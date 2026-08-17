import type { ToolProtocolAdapter } from './base.ts'
import type { NormalizedToolDefinition, ToolParseContext } from '../types.ts'
import {
  buildToolCall,
  createParseResult,
  decodeXml,
  detectMarkers,
  escapeXmlAttribute,
  renderToolList,
  toolNames,
} from './shared.ts'

const DSML = '｜DSML｜'
const TOOL_START = `<${DSML}tool_calls>`
const TOOL_END = `</${DSML}tool_calls>`
const INVOKE_END = `</${DSML}invoke>`
const PARAM_END = `</${DSML}parameter>`
const TAG_NAME_PATTERN = /<\s*(\/?)\s*([^\s>/]+)([^>]*)>/g
const DSML_LOOKALIKE_PATTERN = /<\s*\/?[^\s>]*DSML[^\s>]*/

export const deepseekDsmlProtocol: ToolProtocolAdapter = {
  id: 'deepseek_dsml',

  renderPrompt(tools) {
    return `## Tools

You have access to a set of tools to help answer the user's question.
You can invoke tools by writing a "<${DSML}tool_calls>" block like the following:

<${DSML}tool_calls>
<${DSML}invoke name="$TOOL_NAME">
<${DSML}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</${DSML}parameter>
...
</${DSML}invoke>
<${DSML}invoke name="$TOOL_NAME2">
...
</${DSML}invoke>
</${DSML}tool_calls>

String parameters should be specified as is and set \`string="true"\`.
For all other types (numbers, booleans, arrays, objects), pass the value
in JSON format and set \`string="false"\`.

### Available Tool Schemas

${renderToolList(tools)}

You MUST strictly follow the above defined tool name and parameter schemas
to invoke tool calls.`
  },

  detectStart(buffer) {
    return detectMarkers(buffer, [
      TOOL_START,
      `<${DSML}invoke`,
      '<DSML',
      '<｜｜DSML｜｜tool_calls>',
      '<｜｜DSML｜｜invoke',
    ])
  },

  parse(content: string, context: ToolParseContext) {
    const allowedNames = toolNames(context.tools)
    const rawMatches: string[] = []
    const invalidToolNames: string[] = []
    const toolCalls = []
    const blocks = findElements(content, 'tool_calls', { allowPlain: false })

    if (blocks === 'malformed') return malformedDsmlResult(content, rawMatches, invalidToolNames)

    if (blocks.length === 0) {
      if (looksLikeDsml(content)) {
        if (hasTagNamed(content, 'tool_calls')) {
          return malformedDsmlResult(content, rawMatches, invalidToolNames)
        }
        const recovered = parseInvokes(
          content,
          context,
          rawMatches,
          invalidToolNames,
          allowedNames,
          { allowPlain: false },
        )
        if (recovered === 'malformed') {
          return malformedDsmlResult(content, rawMatches, invalidToolNames)
        }
        if (recovered.length > 0) {
          const cleanContent = removeSpans(content, recovered.map((call) => call.rawText ?? '')).trim()
          return createParseResult({
            content: cleanContent,
            toolCalls: recovered,
            protocol: 'deepseek_dsml',
            rawMatches,
            invalidToolNames,
          })
        }
        return malformedDsmlResult(content, rawMatches, invalidToolNames)
      }
      return createParseResult({ content, toolCalls: [], protocol: 'unknown', rawMatches, invalidToolNames })
    }

    for (const block of blocks) {
      if (block.attrs.trim()) return malformedDsmlResult(content, rawMatches, invalidToolNames)
      rawMatches.push(block.raw)
      const parsed = parseInvokes(block.body, context, rawMatches, invalidToolNames, allowedNames, { allowPlain: true })
      if (parsed === 'malformed') {
        return malformedDsmlResult(content, rawMatches, invalidToolNames)
      }
      toolCalls.push(...parsed)
    }

    if (toolCalls.length === 0) {
      return createParseResult({
        content,
        toolCalls,
        protocol: 'deepseek_dsml',
        rawMatches,
        invalidToolNames,
        malformedReason: rawMatches.length > 0 ? 'DeepSeek DSML block contained no valid invokes' : undefined,
      })
    }

    const cleanContent = removeSpans(content, blocks.map((block) => block.raw)).trim()
    return createParseResult({
      content: cleanContent,
      toolCalls,
      protocol: 'deepseek_dsml',
      rawMatches,
      invalidToolNames,
    })
  },

  formatAssistantToolCalls(calls) {
    const invokes = calls.map((call) => {
      const args = safeParseObject(call.arguments)
      const params = Object.entries(args)
        .map(([name, value]) => formatParameter(name, value))
        .join('')
      return `<${DSML}invoke name="${escapeXmlAttribute(call.name)}">${params}</${DSML}invoke>`
    })
    return `${TOOL_START}${invokes.join('')}${TOOL_END}`
  },

  formatToolResult(result) {
    return `<tool_result>${escapeText(result.content)}</tool_result>`
  },
}

function malformedDsmlResult(
  content: string,
  rawMatches: string[],
  invalidToolNames: string[],
) {
  return createParseResult({
    content,
    toolCalls: [],
    protocol: 'deepseek_dsml',
    rawMatches,
    invalidToolNames,
    malformedReason: 'Malformed DeepSeek DSML tool call block',
  })
}

function parseInvokes(
  content: string,
  context: ToolParseContext,
  rawMatches: string[],
  invalidToolNames: string[],
  allowedNames: Set<string>,
  options: { allowPlain: boolean },
): ReturnType<typeof buildToolCall>[] | 'malformed' {
  const invokes = findElements(content, 'invoke', options)
  if (invokes === 'malformed' || invokes.length === 0) return 'malformed'

  const toolCalls: ReturnType<typeof buildToolCall>[] = []
  for (const invoke of invokes) {
    const attrs = parseAttrs(invoke.attrs)
    const name = attrs.name
    if (!name) return 'malformed'
    rawMatches.push(invoke.raw)
    if (!allowedNames.has(name)) {
      invalidToolNames.push(name)
      continue
    }
    const args = parseParameters(name, invoke.body, context.tools)
    if (args === null) return 'malformed'
    toolCalls.push(
      buildToolCall(
        `call_${toolCalls.length}`,
        toolCalls.length,
        name,
        JSON.stringify(args),
        invoke.raw,
      ),
    )
  }
  const remainder = removeSpans(content, invokes.map((invoke) => invoke.raw)).trim()
  return remainder ? 'malformed' : toolCalls
}

function parseParameters(
  name: string,
  content: string,
  tools: NormalizedToolDefinition[],
): Record<string, unknown> | null {
  const params = findElements(content, 'parameter', { allowPlain: true })
  if (params === 'malformed') return null

  const args: Record<string, unknown> = {}
  for (const param of params) {
    const attrs = parseAttrs(param.attrs)
    const parameterName = attrs.name
    const stringFlag = attrs.string
    if (!parameterName || (stringFlag !== 'true' && stringFlag !== 'false')) return null
    args[parameterName] = decodeParameter(param.body, stringFlag === 'true')
  }
  const remainder = removeSpans(content, params.map((param) => param.raw)).trim()
  if (remainder) return null
  return repairWrapper(name, args, tools)
}

function parseAttrs(attrs: string): Record<string, string> {
  const parsed: Record<string, string> = {}
  const attrPattern = /([a-zA-Z_][\w:-]*)\s*=\s*(['"])(.*?)\2/g
  for (const match of attrs.matchAll(attrPattern)) {
    parsed[match[1]] = decodeXml(match[3])
  }
  return parsed
}

function decodeParameter(value: string, isString: boolean): unknown {
  const decoded = decodeXml(value)
  if (isString) return decoded
  try {
    return JSON.parse(decoded)
  } catch {
    return decoded
  }
}

function repairWrapper(name: string, args: Record<string, unknown>, tools: NormalizedToolDefinition[]): Record<string, unknown> {
  const keys = Object.keys(args)
  if (keys.length !== 1 || (keys[0] !== 'arguments' && keys[0] !== 'input')) return args
  const tool = tools.find((item) => item.name === name)
  const properties = getProperties(tool?.parameters)
  if (properties && keys[0] in properties) return args
  const wrapped = args[keys[0]]
  if (!wrapped || typeof wrapped !== 'object' || Array.isArray(wrapped)) return args
  const wrappedRecord = wrapped as Record<string, unknown>
  if (properties && !Object.keys(wrappedRecord).every((key) => key in properties)) return args
  return wrappedRecord
}

function getProperties(parameters: Record<string, unknown> | undefined): Record<string, unknown> | null {
  const properties = parameters?.properties
  return properties && typeof properties === 'object' && !Array.isArray(properties)
    ? properties as Record<string, unknown>
    : null
}

function formatParameter(name: string, value: unknown): string {
  const isString = typeof value === 'string'
  const encoded = isString ? value : JSON.stringify(value)
  return `<${DSML}parameter name="${escapeXmlAttribute(name)}" string="${isString ? 'true' : 'false'}">${escapeText(encoded)}</${DSML}parameter>`
}

function safeParseObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : { arguments: parsed }
  } catch {
    return { arguments: value }
  }
}

interface DsmlElement {
  tag: string
  attrs: string
  body: string
  raw: string
}

function findElements(
  content: string,
  semantic: string,
  options: { allowPlain: boolean },
): DsmlElement[] | 'malformed' {
  const elements: DsmlElement[] = []
  const stack: Array<{ tag: string, attrs: string, start: number, bodyStart: number }> = []
  TAG_NAME_PATTERN.lastIndex = 0
  for (const match of content.matchAll(TAG_NAME_PATTERN)) {
    const closing = match[1]
    const tag = match[2]
    const attrs = match[3]
    const semanticMatch = (options.allowPlain && tag === semantic) || isDsmlTag(tag, semantic)
    if (!semanticMatch) continue

    if (closing) {
      const open = stack.pop()
      if (!open || open.tag !== tag) return 'malformed'
      if (stack.length === 0) {
        elements.push({
          tag,
          attrs: open.attrs,
          body: content.slice(open.bodyStart, match.index),
          raw: content.slice(open.start, match.index + match[0].length),
        })
      }
      continue
    }

    if (attrs.trimEnd().endsWith('/')) return 'malformed'
    stack.push({ tag, attrs, start: match.index, bodyStart: match.index + match[0].length })
  }
  return stack.length > 0 ? 'malformed' : elements
}

function isDsmlTag(tag: string, semantic: string): boolean {
  if (!tag.endsWith(semantic)) return false
  const marker = tag.slice(0, -semantic.length)
  if (!marker.includes('DSML')) return false
  return marker
    .split('DSML')
    .every((part) => [...part].every(isMarkerChar))
}

function isMarkerChar(char: string): boolean {
  return /^[\p{P}\p{S}]$/u.test(char)
}

function hasTagNamed(content: string, semantic: string): boolean {
  TAG_NAME_PATTERN.lastIndex = 0
  for (const match of content.matchAll(TAG_NAME_PATTERN)) {
    const tag = match[2]
    if (tag === semantic || isDsmlTag(tag, semantic)) return true
  }
  return false
}

function removeSpans(content: string, rawSpans: string[]): string {
  return rawSpans.reduce((acc, raw) => acc.replace(raw, ''), content)
}

function looksLikeDsml(content: string): boolean {
  return content.includes(`<${DSML}`)
    || content.includes(`</${DSML}`)
    || content.includes('<｜｜DSML｜｜')
    || DSML_LOOKALIKE_PATTERN.test(content)
}

function escapeText(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}
