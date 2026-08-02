/**
 * DeepSeek Adapter
 * Implements DeepSeek web API protocol
 * 
 * NOTE: Tool prompt injection is handled by Forwarder.transformRequestForPromptToolUse()
 * This adapter only handles message format conversion and API communication
 */

import axios, { AxiosResponse } from 'axios'
import FormData from 'form-data'
import mime from 'mime-types'
import path from 'path'
import { getDeepSeekHash } from '../../lib/challenge'
import type { Account, Provider } from '../../store/types'
import type { ChatMessageContent } from '../types'
import { resolveDeepSeekChatOptions } from './providerModelOptions'
import { getProviderToolProfile } from '../toolCalling/providerProfiles'

const DEEPSEEK_API_BASE = 'https://chat.deepseek.com/api'
const CHAT_COMPLETION_PATH = '/api/v0/chat/completion'
const FILE_UPLOAD_PATH = '/api/v0/file/upload_file'
const FILE_FETCH_PATH = '/api/v0/file/fetch_files'
const FILE_MAX_SIZE = 100 * 1024 * 1024 // 100MB
const FILE_POLL_ATTEMPTS = 20
const FILE_POLL_INTERVAL_MS = 1000
const CHALLENGE_RETRY_ATTEMPTS = 2

const FAKE_HEADERS = {
  Accept: '*/*',
  'Accept-Encoding': 'gzip, deflate, br, zstd',
  'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
  Origin: 'https://chat.deepseek.com',
  Referer: 'https://chat.deepseek.com/',
  'Sec-Ch-Ua': '"Not/A)Brand";v="99", "Chromium";v="148"',
  'Sec-Ch-Ua-Mobile': '?0',
  'Sec-Ch-Ua-Platform': '"macOS"',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-origin',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
  'X-App-Version': '2.0.0',
  'X-Client-Locale': 'zh_CN',
  'X-Client-Platform': 'web',
  'x-Client-Timezone-Offset': '28800',
  'X-Client-Version': '2.0.0',
}

interface TokenInfo {
  accessToken: string
  refreshToken: string
  expiresAt: number
}

interface ChallengeResponse {
  algorithm: string
  challenge: string
  salt: string
  difficulty: number
  expire_at: number
  signature: string
}

interface DeepSeekMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  content: string | ChatMessageContent[] | null
  tool_call_id?: string
  tool_calls?: any[]
}

interface DeepSeekImageFile {
  url: string
  filename: string
  mimeType: string
  data: Buffer
}

interface DeepSeekUploadedFile {
  id: string
  status?: string
  audit_result?: string
  model_kind?: string
  is_image?: boolean
  [key: string]: any
}

interface ChatCompletionRequest {
  model: string
  messages: DeepSeekMessage[]
  stream?: boolean
  temperature?: number
  web_search?: boolean
  reasoning_effort?: 'low' | 'medium' | 'high'
  tools?: any[]
  tool_choice?: any
}

const tokenCache = new Map<string, TokenInfo>()
const sessionCache = new Map<string, { sessionId: string; createdAt: number }>()

function generateRandomString(length: number, charset: string = 'alphanumeric'): string {
  const sets = {
    numeric: '0123456789',
    alphabetic: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    alphanumeric: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
    hex: '0123456789abcdef',
  }
  const chars = sets[charset as keyof typeof sets] || sets.alphanumeric
  let result = ''
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

function uuid(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

function generateCookie(): string {
  const timestamp = Date.now()
  return `intercom-HWWAFSESTIME=${timestamp}; HWWAFSESID=${generateRandomString(18, 'hex')}; Hm_lvt_${uuid()}=${Math.floor(timestamp / 1000)},${Math.floor(timestamp / 1000)},${Math.floor(timestamp / 1000)}; Hm_lpvt_${uuid()}=${Math.floor(timestamp / 1000)}; _frid=${uuid()}; _fr_ssid=${uuid()}; _fr_pvid=${uuid()}`
}

function unixTimestamp(): number {
  return Math.floor(Date.now() / 1000)
}

function deepSeekApiUrl(targetPath: string): string {
  return `${DEEPSEEK_API_BASE}${targetPath.replace(/^\/api/, '')}`
}

export class DeepSeekAdapter {
  private provider: Provider
  private account: Account
  private token: string

  constructor(provider: Provider, account: Account) {
    this.provider = provider
    this.account = account
    this.token = account.credentials.token || account.credentials.apiKey || account.credentials.refreshToken || ''
    console.log('[DeepSeek] Adapter initialized for account:', account.id)
  }

  private async acquireToken(): Promise<string> {
    if (!this.token) {
      throw new Error('DeepSeek Token not configured, please add Token in account settings')
    }

    const cached = tokenCache.get(this.token)
    if (cached && cached.expiresAt > unixTimestamp()) {
      return cached.accessToken
    }

    console.log('[DeepSeek] Acquiring token...')
    
    const result = await axios.get(`${DEEPSEEK_API_BASE}/v0/users/current`, {
      headers: {
        Authorization: `Bearer ${this.token}`,
        ...FAKE_HEADERS,
      },
      timeout: 15000,
      validateStatus: () => true,
    })

    console.log('[DeepSeek] Token response status:', result.status)
    
    if (result.status === 401 || result.status === 403) {
      throw new Error(`Token invalid or expired, please get a new Token`)
    }

    if (result.status !== 200) {
      throw new Error(`Failed to acquire token: HTTP ${result.status}`)
    }

    // Response structure: { code: 0, data: { biz_code: 0, biz_data: { token: "..." } } }
    const bizData = result.data?.data?.biz_data || result.data?.biz_data
    if (!bizData?.token) {
      const errorMsg = result.data?.msg || result.data?.data?.biz_msg || 'Unknown error'
      console.log('[DeepSeek] Token response missing access token')
      throw new Error(`Failed to acquire token: ${errorMsg}`)
    }

    const accessToken = bizData.token
    tokenCache.set(this.token, {
      accessToken,
      refreshToken: this.token,
      expiresAt: unixTimestamp() + 3600,
    })

    console.log('[DeepSeek] Token acquired successfully')
    return accessToken
  }

  private async createSession(): Promise<string> {
    const cacheKey = this.account.id
    const cached = sessionCache.get(cacheKey)
    if (cached && Date.now() - cached.createdAt < 300000) {
      return cached.sessionId
    }

    const token = await this.acquireToken()
    const result = await axios.post(
      `${DEEPSEEK_API_BASE}/v0/chat_session/create`,
      {},
      {
        headers: {
          Authorization: `Bearer ${token}`,
          ...FAKE_HEADERS,
          Cookie: generateCookie(),
        },
        timeout: 15000,
        validateStatus: () => true,
      }
    )

    console.log('[DeepSeek] Create session response status:', result.status)

    // Older responses wrap the session in `chat_session`; current responses expose it directly.
    const bizData = result.data?.data?.biz_data || result.data?.biz_data
    const sessionId = bizData?.chat_session?.id || bizData?.id
    if (result.status !== 200 || !sessionId) {
      throw new Error(`Failed to create session: ${result.data?.msg || result.data?.data?.biz_msg || result.status}`)
    }

    console.log('[DeepSeek] Created session')
    sessionCache.set(cacheKey, { sessionId, createdAt: Date.now() })

    return sessionId
  }

  async deleteSession(sessionId: string): Promise<boolean> {
    try {
      const token = await this.acquireToken()
      const result = await axios.post(
        `${DEEPSEEK_API_BASE}/v0/chat_session/delete`,
        { chat_session_id: sessionId },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            ...FAKE_HEADERS,
          },
          timeout: 15000,
          validateStatus: () => true,
        }
      )

      console.log('[DeepSeek] Delete session response:', JSON.stringify(result.data, null, 2))

      const success = result.status === 200 && result.data?.code === 0
      if (success) {
        // Clear cache
        const cacheKey = this.account.id
        sessionCache.delete(cacheKey)
        console.log('[DeepSeek] Session deleted')
      }
      return success
    } catch (error) {
      console.error('[DeepSeek] Failed to delete session:', error)
      return false
    }
  }

  private async getChallenge(targetPath: string): Promise<ChallengeResponse> {
    const token = await this.acquireToken()
    let lastError: unknown

    for (let attempt = 1; attempt <= CHALLENGE_RETRY_ATTEMPTS; attempt++) {
      try {
        const result = await axios.post(
          `${DEEPSEEK_API_BASE}/v0/chat/create_pow_challenge`,
          { target_path: targetPath },
          {
            headers: {
              Authorization: `Bearer ${token}`,
              ...FAKE_HEADERS,
            },
            timeout: 30000,
            validateStatus: () => true,
          }
        )

        // Response structure: { code: 0, data: { biz_code: 0, biz_data: { challenge: {...} } } }
        const bizData = result.data?.data?.biz_data || result.data?.biz_data
        if (result.status !== 200 || !bizData?.challenge) {
          throw new Error(`Failed to get challenge: ${result.data?.msg || result.data?.data?.biz_msg || result.status}`)
        }

        return bizData.challenge
      } catch (error) {
        lastError = error
        if (attempt < CHALLENGE_RETRY_ATTEMPTS) {
          console.log('[DeepSeek] Retrying challenge request:', { targetPath, attempt })
          await this.sleep(1000)
        }
      }
    }

    throw lastError
  }

  private async calculateChallengeAnswer(challenge: ChallengeResponse, targetPath: string): Promise<string> {
    const { algorithm, challenge: challengeStr, salt, difficulty, expire_at, signature } = challenge

    if (algorithm !== 'DeepSeekHashV1') {
      throw new Error(`Unsupported algorithm: ${algorithm}`)
    }

    console.log('[DeepSeek] Challenge parameters:', { difficulty, targetPath })

    const deepSeekHash = await getDeepSeekHash()
    const answer = deepSeekHash.calculateHash(algorithm, challengeStr, salt, difficulty, expire_at)

    if (answer === undefined) {
      throw new Error('Challenge calculation failed')
    }

    console.log('[DeepSeek] Challenge answer calculated')

    return Buffer.from(JSON.stringify({
      algorithm,
      challenge: challengeStr,
      salt,
      answer,
      signature,
      target_path: targetPath,
    })).toString('base64')
  }

  private isBase64Data(url: string): boolean {
    return /^data:[^;]+;base64,/i.test(url)
  }

  private extractBase64Format(url: string): string {
    const match = url.match(/^data:([^;]+);base64,/i)
    return match ? match[1] : 'application/octet-stream'
  }

  private removeBase64Header(url: string): string {
    return url.replace(/^data:[^;]+;base64,/i, '')
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  private extractImageUrls(messages: DeepSeekMessage[]): string[] {
    const imageUrls: string[] = []

    for (const message of messages) {
      if (!Array.isArray(message.content)) continue

      for (const part of message.content) {
        if (part.type !== 'image_url') continue

        const imageUrl = typeof (part as any).image_url === 'string'
          ? (part as any).image_url
          : part.image_url?.url

        if (imageUrl) {
          imageUrls.push(imageUrl)
        }
      }
    }

    return imageUrls
  }

  private async loadImageFile(fileUrl: string): Promise<DeepSeekImageFile> {
    let filename: string
    let data: Buffer
    let mimeType: string

    if (fileUrl.startsWith('data:')) {
      if (!this.isBase64Data(fileUrl)) {
        throw new Error('DeepSeek vision data URL must be base64 encoded')
      }

      mimeType = this.extractBase64Format(fileUrl)
      const ext = mime.extension(mimeType) || 'bin'
      filename = `${uuid()}.${ext}`
      data = Buffer.from(this.removeBase64Header(fileUrl), 'base64')
    } else {
      let parsedUrl: URL
      try {
        parsedUrl = new URL(fileUrl)
      } catch {
        throw new Error('DeepSeek vision image_url must be a data URL or absolute HTTP(S) URL')
      }

      if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
        throw new Error('DeepSeek vision remote image_url must use HTTP(S)')
      }

      filename = path.basename(parsedUrl.pathname) || `${uuid()}.bin`
      const response = await axios.get(fileUrl, {
        responseType: 'arraybuffer',
        maxContentLength: FILE_MAX_SIZE,
        timeout: 60000,
        validateStatus: () => true,
      })

      if (response.status < 200 || response.status >= 300) {
        throw new Error(`Failed to download DeepSeek vision image: HTTP ${response.status}`)
      }

      data = Buffer.from(response.data)
      const contentType = response.headers['content-type']
      mimeType = (Array.isArray(contentType) ? contentType[0] : contentType)
        || mime.lookup(filename)
        || 'application/octet-stream'
    }

    if (!mimeType.toLowerCase().startsWith('image/')) {
      throw new Error(`DeepSeek vision only supports image content, got ${mimeType}`)
    }

    if (data.length > FILE_MAX_SIZE) {
      throw new Error(`DeepSeek vision image exceeds ${FILE_MAX_SIZE} bytes`)
    }

    return { url: fileUrl, filename, mimeType, data }
  }

  private extractUploadedFile(responseData: any): DeepSeekUploadedFile | null {
    const bizData = responseData?.data?.biz_data || responseData?.biz_data || responseData?.data || responseData
    const candidates = [
      bizData?.file,
      bizData?.file_info,
      bizData?.fileInfo,
      bizData,
    ]

    for (const files of [bizData?.files, bizData?.file_infos, bizData?.fileInfos]) {
      if (Array.isArray(files)) {
        candidates.push(...files)
      }
    }

    for (const candidate of candidates) {
      if (candidate?.id && typeof candidate.id === 'string') {
        return candidate
      }
    }

    return null
  }

  private extractFetchedFiles(responseData: any): DeepSeekUploadedFile[] {
    const bizData = responseData?.data?.biz_data || responseData?.biz_data || responseData?.data || responseData
    if (bizData?.id && typeof bizData.id === 'string') return [bizData]

    const files = bizData?.files || bizData?.file_infos || bizData?.fileInfos || bizData

    if (Array.isArray(files)) return files
    if (files && typeof files === 'object') {
      return Object.values(files).filter((file: any) => file?.id && typeof file.id === 'string') as DeepSeekUploadedFile[]
    }
    return []
  }

  private isUploadedFileReady(file: DeepSeekUploadedFile): boolean {
    const status = String(file.status || '').toUpperCase()
    const auditResult = String(file.audit_result || file.auditResult || '').toLowerCase()
    const modelKind = String(file.model_kind || file.modelKind || '').toUpperCase()

    return status === 'SUCCESS'
      && (!auditResult || auditResult === 'pass')
      && (!modelKind || modelKind === 'VISION')
      && file.is_image !== false
  }

  private async uploadImageFile(imageFile: DeepSeekImageFile, token: string): Promise<string> {
    console.log('[DeepSeek] Uploading vision image:', {
      filename: imageFile.filename,
      mimeType: imageFile.mimeType,
      size: imageFile.data.length,
    })

    const challenge = await this.getChallenge(FILE_UPLOAD_PATH)
    const challengeAnswer = await this.calculateChallengeAnswer(challenge, FILE_UPLOAD_PATH)

    const formData = new FormData()
    formData.append('file', imageFile.data, {
      filename: imageFile.filename,
      contentType: imageFile.mimeType,
    })

    const response = await axios.post(
      deepSeekApiUrl(FILE_UPLOAD_PATH),
      formData,
      {
        headers: {
          Authorization: `Bearer ${token}`,
          ...FAKE_HEADERS,
          ...formData.getHeaders(),
          Cookie: generateCookie(),
          Referer: 'https://chat.deepseek.com/',
          'X-Ds-Pow-Response': challengeAnswer,
          'X-File-Size': String(imageFile.data.length),
          'X-Model-Type': 'vision',
        },
        maxBodyLength: FILE_MAX_SIZE,
        timeout: 60000,
        validateStatus: () => true,
      }
    )

    const uploadedFile = this.extractUploadedFile(response.data)
    if (response.status !== 200 || !uploadedFile?.id) {
      const errorMsg = response.data?.msg || response.data?.data?.biz_msg || `HTTP ${response.status}`
      throw new Error(`DeepSeek vision upload failed: ${errorMsg}`)
    }

    console.log('[DeepSeek] Vision image uploaded')
    await this.waitForUploadedFile(uploadedFile.id, token)
    return uploadedFile.id
  }

  private async waitForUploadedFile(fileId: string, token: string): Promise<DeepSeekUploadedFile> {
    for (let attempt = 1; attempt <= FILE_POLL_ATTEMPTS; attempt++) {
      const response = await axios.get(
        deepSeekApiUrl(FILE_FETCH_PATH),
        {
          params: { file_ids: fileId },
          headers: {
            Authorization: `Bearer ${token}`,
            ...FAKE_HEADERS,
            Cookie: generateCookie(),
            Referer: 'https://chat.deepseek.com/',
          },
          timeout: 15000,
          validateStatus: () => true,
        }
      )

      if (response.status !== 200) {
        throw new Error(`Failed to fetch DeepSeek vision file status: HTTP ${response.status}`)
      }

      const file = this.extractFetchedFiles(response.data).find((item) => item.id === fileId)
      if (!file) {
        throw new Error(`DeepSeek vision file not found after upload: ${fileId}`)
      }

      const status = String(file.status || '').toUpperCase()
      if (this.isUploadedFileReady(file)) {
        console.log('[DeepSeek] Vision image ready')
        return file
      }

      if (['FAILED', 'FAIL', 'ERROR'].includes(status)) {
        throw new Error(`DeepSeek vision file processing failed: ${status}`)
      }

      const auditResult = String(file.audit_result || file.auditResult || '').toLowerCase()
      if (['failed', 'fail', 'reject', 'rejected', 'block', 'blocked', 'not_pass'].includes(auditResult)) {
        throw new Error(`DeepSeek vision file audit failed: ${auditResult}`)
      }

      console.log('[DeepSeek] Waiting for vision image processing:', { status, auditResult, attempt })
      await this.sleep(FILE_POLL_INTERVAL_MS)
    }

    throw new Error(`DeepSeek vision file processing timed out: ${fileId}`)
  }

  private async uploadImages(imageUrls: string[], token: string): Promise<string[]> {
    const refFileIds: string[] = []

    for (const imageUrl of imageUrls) {
      const imageFile = await this.loadImageFile(imageUrl)
      const fileId = await this.uploadImageFile(imageFile, token)
      refFileIds.push(fileId)
    }

    return refFileIds
  }

  private messagesToPrompt(messages: DeepSeekMessage[], isMultiTurn: boolean = false): string {
    const toolProfile = getProviderToolProfile('deepseek')
    const processedMessages = messages.map(message => {
      let text: string

      // Handle tool calls in assistant message
      if (message.role === 'assistant' && message.tool_calls && message.tool_calls.length > 0) {
        text = toolProfile.formatAssistantToolCalls(message.tool_calls.map(tc => ({
          id: tc.id,
          name: tc.function.name,
          arguments: tc.function.arguments,
        })))
      }
      // Handle tool response message
      else if (message.role === 'tool' && message.tool_call_id) {
        text = toolProfile.formatToolResult({
          toolCallId: message.tool_call_id,
          content: String(message.content || ''),
        })
      }
      else if (Array.isArray(message.content)) {
        const texts = message.content
          .filter((item: any) => item.type === 'text')
          .map((item: any) => item.text)
        text = texts.join('\n')
      } else {
        text = String(message.content || '')
      }
      return { role: message.role, text }
    })

    if (processedMessages.length === 0) return ''

    // For multi-turn mode, only send the last user message
    if (isMultiTurn) {
      let lastUserIdx = -1
      for (let i = processedMessages.length - 1; i >= 0; i--) {
        if (processedMessages[i].role === 'user') {
          lastUserIdx = i
          break
        }
      }
      
      if (lastUserIdx !== -1) {
        const lastUserMsg = processedMessages[lastUserIdx]
        let text = lastUserMsg.text
        for (let i = lastUserIdx + 1; i < processedMessages.length; i++) {
          if (processedMessages[i].role === 'tool') {
            text += `\n\n${processedMessages[i].text}`
          }
        }
        return `<｜User｜>${text}`
      }
    }

    const mergedBlocks: { role: string; text: string }[] = []
    let currentBlock = { ...processedMessages[0] }

    for (let i = 1; i < processedMessages.length; i++) {
      const msg = processedMessages[i]
      if (msg.role === currentBlock.role) {
        currentBlock.text += `\n\n${msg.text}`
      } else {
        mergedBlocks.push(currentBlock)
        currentBlock = { ...msg }
      }
    }
    mergedBlocks.push(currentBlock)

    return mergedBlocks
      .map((block, index) => {
        if (block.role === 'assistant') {
          return `<｜Assistant｜>${block.text}<｜end of sentence｜>`
        }
        if (block.role === 'user' || block.role === 'system') {
          return index > 0 ? `<｜User｜>${block.text}` : block.text
        }
        if (block.role === 'tool') {
          return `<｜User｜>${block.text}`
        }
        return block.text
      })
      .join('')
      .replace(/!\[.+\]\(.+\)/g, '')
  }

  async chatCompletion(request: ChatCompletionRequest): Promise<{ response: AxiosResponse; sessionId: string }> {
    const token = await this.acquireToken()

    const sessionId = await this.createSession()
    console.log('[DeepSeek] Using chat session')

    // Clone messages to avoid modifying original request
    // Note: Tool prompt injection is already handled by Forwarder.transformRequestForPromptToolUse()
    const messages = [...request.messages]
    const imageUrls = this.extractImageUrls(messages)
    const refFileIds = imageUrls.length > 0
      ? await this.uploadImages(imageUrls, token)
      : []

    const challenge = await this.getChallenge(CHAT_COMPLETION_PATH)
    const challengeAnswer = await this.calculateChallengeAnswer(challenge, CHAT_COMPLETION_PATH)

    const prompt = this.messagesToPrompt(messages, false)

    const { modelType, searchEnabled, thinkingEnabled } = resolveDeepSeekChatOptions(request, prompt)
    const finalModelType = refFileIds.length > 0 ? 'vision' : modelType

    if (request.web_search || request.model.toLowerCase().includes('search')) {
      console.log('[DeepSeek] Web search enabled')
    }

    if (request.reasoning_effort || thinkingEnabled) {
      console.log('[DeepSeek] Reasoning mode enabled, effort:', request.reasoning_effort)
    }

    const response = await axios.post(
      deepSeekApiUrl(CHAT_COMPLETION_PATH),
      {
        chat_session_id: sessionId,
        parent_message_id: null,
        prompt,
        model_type: finalModelType,
        ref_file_ids: refFileIds,
        search_enabled: searchEnabled,
        thinking_enabled: thinkingEnabled,
        preempt: false,
      },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          ...FAKE_HEADERS,
          Referer: `https://chat.deepseek.com/a/chat/s/${sessionId}`,
          Cookie: generateCookie(),
          'X-Ds-Pow-Response': challengeAnswer,
        },
        timeout: 120000,
        validateStatus: () => true,
        responseType: 'stream',
      }
    )

    return { response, sessionId }
  }

  async deleteAllChats(): Promise<boolean> {
    try {
      const token = await this.acquireToken()
      const result = await axios.post(
        `${DEEPSEEK_API_BASE}/v0/chat_session/delete_all`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
            ...FAKE_HEADERS,
          },
          timeout: 30000,
          validateStatus: () => true,
        }
      )

      console.log('[DeepSeek] Delete all chats response:', JSON.stringify(result.data, null, 2))

      const success = result.status === 200 && result.data?.code === 0
      if (success) {
        sessionCache.clear()
        console.log('[DeepSeek] All chats deleted')
      }
      return success
    } catch (error) {
      console.error('[DeepSeek] Failed to delete all chats:', error)
      return false
    }
  }

  static isDeepSeekProvider(provider: Provider): boolean {
    return provider.id === 'deepseek' || provider.apiEndpoint.includes('deepseek.com')
  }

  /**
   * Clear session cache for a specific account
   * This should be called when a session is deleted externally (e.g., from web)
   */
  static clearSessionCache(accountId: string): void {
    sessionCache.delete(accountId)
    console.log('[DeepSeek] Cleared session cache for account:', accountId)
  }
}

export const deepSeekAdapter = {
  DeepSeekAdapter,
}
