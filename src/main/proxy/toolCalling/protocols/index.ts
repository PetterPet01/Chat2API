import type { ToolProtocolAdapter } from './base.ts'
import type { ToolProtocolId } from '../types.ts'
import { managedBracketProtocol } from './managedBracket.ts'
import { managedXmlProtocol } from './managedXml.ts'
import { deepseekDsmlProtocol } from './deepseekDsml.ts'
import { anthropicToolUseProtocol } from './anthropicToolUse.ts'
import { codexResponsesProtocol } from './codexResponses.ts'

const protocols: Record<ToolProtocolId, ToolProtocolAdapter> = {
  openai_chat: managedBracketProtocol,
  managed_bracket: managedBracketProtocol,
  managed_xml: managedXmlProtocol,
  deepseek_dsml: deepseekDsmlProtocol,
  anthropic_tool_use: anthropicToolUseProtocol,
  codex_responses: codexResponsesProtocol,
}

export function getToolProtocol(id: ToolProtocolId): ToolProtocolAdapter {
  return protocols[id]
}

export function getManagedProtocols(): ToolProtocolAdapter[] {
  return [
    managedBracketProtocol,
    managedXmlProtocol,
    deepseekDsmlProtocol,
    anthropicToolUseProtocol,
    codexResponsesProtocol,
  ]
}
