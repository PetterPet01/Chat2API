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
    return detectMarkers(buffer, [TOOL_START, `<${DSML}invoke`, '<｜｜DSML｜｜tool_calls>', '<｜｜DSML｜｜invoke'])
  },

  parse(content: string, context: ToolParseContext) {
    const allowedNames = toolNames(context.tools)
    const rawMatches: string[] = []
    const invalidToolNames: string[] = []
    const toolCalls = []
    const blocks = [...content.matchAll(new RegExp(`<${DSML}tool_calls>([\\s\\S]*?)</${DSML}tool_calls>`, 'g'))]

    if (blocks.length === 0) {
      if (looksLikeDsml(content)) {
        const recovered = parseInvokes(content.replaceAll('｜｜DSML｜｜', DSML), context, rawMatches, invalidToolNames, allowedNames)
        if (recovered === 'malformed') {
          return createParseResult({
            content,
            toolCalls: [],
            protocol: 'deepseek_dsml',
            rawMatches,
            invalidToolNames,
            malformedReason: 'Malformed DeepSeek DSML tool call block',
          })
        }
        if (toolCalls.length === 0 && recovered.length > 0) {
          const cleanContent = recovered.reduce((acc, raw) => acc.replace(raw, ''), content.replaceAll('｜｜DSML｜｜', DSML)).trim()
          return createParseResult({
            content: cleanContent,
            toolCalls: recovered,
            protocol: 'deepseek_dsml',
            rawMatches,
            invalidToolNames,
          })
        }
        return createParseResult({
          content,
          toolCalls: [],
          protocol: 'deepseek_dsml',
          rawMatches,
          invalidToolNames,
          malformedReason: 'Malformed DeepSeek DSML tool call block',
        })
      }
      return createParseResult({ content, toolCalls: [], protocol: 'unknown', rawMatches, invalidToolNames })
    }

    for (const block of blocks) {
      rawMatches.push(block[0])
      const parsed = parseInvokes(block[1], context, rawMatches, invalidToolNames, allowedNames)
      if (parsed === 'malformed') {
        return createParseResult({
          content,
          toolCalls: [],
          protocol: 'deepseek_dsml',
          rawMatches,
          invalidToolNames,
          malformedReason: 'Malformed DeepSeek DSML tool call block',
        })
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

    const cleanContent = blocks.reduce((acc, block) => acc.replace(block[0], ''), content).trim()
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

function parseInvokes(
  content: string,
  context: ToolParseContext,
  rawMatches: string[],
  invalidToolNames: string[],
  allowedNames: Set<string>,
): ReturnType<typeof buildToolCall>[] | 'malformed' {
  const invokePattern = new RegExp(`<${DSML}invoke\\s+name="([^"]+)"\\s*>([\\s\\S]*?)</${DSML}invoke>`, 'g')
  const invokes = [...content.matchAll(invokePattern)]
  if (invokes.length === 0) return 'malformed'

  const toolCalls: ReturnType<typeof buildToolCall>[] = []
  for (const invoke of invokes) {
    const name = decodeXml(invoke[1].trim())
    rawMatches.push(invoke[0])
    if (!allowedNames.has(name)) {
      invalidToolNames.push(name)
      continue
    }
    const args = parseParameters(name, invoke[2], context.tools)
    if (args === null) return 'malformed'
    toolCalls.push(
      buildToolCall(
        `call_${toolCalls.length}`,
        toolCalls.length,
        name,
        JSON.stringify(args),
        invoke[0],
      ),
    )
  }
  const remainder = content.replace(invokePattern, '').trim()
  return remainder ? 'malformed' : toolCalls
}

function parseParameters(name: string, content: string, tools: NormalizedToolDefinition[]): Record<string, unknown> | null {
  const parameterPattern = new RegExp(
    `<${DSML}parameter\\s+([^>]*)>([\\s\\S]*?)</${DSML}parameter>`,
    'g',
  )
  const args: Record<string, unknown> = {}
  for (const match of content.matchAll(parameterPattern)) {
    const attrs = parseAttrs(match[1])
    const parameterName = attrs.name
    const stringFlag = attrs.string
    if (!parameterName || (stringFlag !== 'true' && stringFlag !== 'false')) return null
    args[parameterName] = decodeParameter(match[2], stringFlag === 'true')
  }
  const remainder = content.replace(parameterPattern, '').trim()
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

function looksLikeDsml(content: string): boolean {
  return content.includes(`<${DSML}`) || content.includes(`</${DSML}`) || content.includes('<｜｜DSML｜｜')
}

function escapeText(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}
