import test from 'node:test'
import assert from 'node:assert/strict'
import { managedBracketProtocol } from '../../src/main/proxy/toolCalling/protocols/managedBracket.ts'
import { managedXmlProtocol } from '../../src/main/proxy/toolCalling/protocols/managedXml.ts'
import { deepseekDsmlProtocol } from '../../src/main/proxy/toolCalling/protocols/deepseekDsml.ts'
import { anthropicToolUseProtocol } from '../../src/main/proxy/toolCalling/protocols/anthropicToolUse.ts'
import { codexResponsesProtocol } from '../../src/main/proxy/toolCalling/protocols/codexResponses.ts'

const tools = [
  {
    name: 'default_api:read_file',
    description: 'Read a file',
    parameters: {
      type: 'object',
      properties: {
        filePath: { type: 'string' },
        lineCount: { type: 'integer' },
      },
    },
    source: 'openai' as const,
  },
]

test('managed bracket parses valid tool call', () => {
  const result = managedBracketProtocol.parse(
    '[function_calls]\n[call:default_api:read_file]{"filePath":"/tmp/a"}[/call]\n[/function_calls]',
    { tools, protocol: 'managed_bracket' },
  )

  assert.equal(result.toolCalls.length, 1)
  assert.equal(result.toolCalls[0].function.name, 'default_api:read_file')
  assert.equal(result.content, '')
})

test('managed xml parses valid Chat2API tool call', () => {
  const result = managedXmlProtocol.parse(
    '<|CHAT2API|tool_calls><|CHAT2API|invoke name="default_api:read_file"><|CHAT2API|parameter name="filePath"><![CDATA[/tmp/a]]></|CHAT2API|parameter></|CHAT2API|invoke></|CHAT2API|tool_calls>',
    { tools, protocol: 'managed_xml' },
  )

  assert.equal(result.toolCalls.length, 1)
  assert.equal(result.toolCalls[0].function.name, 'default_api:read_file')
})

test('managed xml parses canonical XML compatibility form', () => {
  const result = managedXmlProtocol.parse(
    '<tool_calls><invoke name="default_api:read_file"><parameter name="filePath">/tmp/a</parameter></invoke></tool_calls>',
    { tools, protocol: 'managed_xml' },
  )

  assert.equal(result.toolCalls.length, 1)
  assert.equal(JSON.parse(result.toolCalls[0].function.arguments).filePath, '/tmp/a')
})

test('deepseek dsml parses native typed tool call and strips native syntax', () => {
  const result = deepseekDsmlProtocol.parse(
    'Need data<｜DSML｜tool_calls><｜DSML｜invoke name="default_api:read_file"><｜DSML｜parameter name="filePath" string="true">/tmp/a</｜DSML｜parameter><｜DSML｜parameter name="lineCount" string="false">3</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>',
    { tools, protocol: 'deepseek_dsml' },
  )

  assert.equal(result.protocol, 'deepseek_dsml')
  assert.equal(result.toolCalls.length, 1)
  assert.equal(result.toolCalls[0].function.name, 'default_api:read_file')
  assert.deepEqual(JSON.parse(result.toolCalls[0].function.arguments), {
    filePath: '/tmp/a',
    lineCount: 3,
  })
  assert.equal(result.content, 'Need data')
  assert.equal(result.content.includes('｜DSML｜'), false)
})

test('deepseek dsml parses DSH wrapper-dialect tool calls', () => {
  const result = deepseekDsmlProtocol.parse(
    'Read context first<DSML｜tool_calls><invoke name="default_api:read_file"><parameter name="filePath" string="true">/tmp/task_plan.md</parameter></invoke><invoke name="default_api:read_file"><parameter name="filePath" string="true">/tmp/findings.md</parameter></invoke></DSML｜tool_calls>',
    { tools, protocol: 'deepseek_dsml' },
  )

  assert.equal(result.protocol, 'deepseek_dsml')
  assert.equal(result.toolCalls.length, 2)
  assert.deepEqual(
    result.toolCalls.map((call) => JSON.parse(call.function.arguments)),
    [{ filePath: '/tmp/task_plan.md' }, { filePath: '/tmp/findings.md' }],
  )
  assert.equal(result.content, 'Read context first')
  assert.equal(result.content.includes('DSML｜'), false)
})

test('deepseek dsml rejects malformed DSH wrapper-dialect output explicitly', () => {
  const result = deepseekDsmlProtocol.parse(
    '<DSML｜tool_calls><invoke name="default_api:read_file"><parameter name="filePath" string="true">/tmp/a</invoke></DSML｜tool_calls>',
    { tools, protocol: 'deepseek_dsml' },
  )

  assert.equal(result.protocol, 'deepseek_dsml')
  assert.equal(result.toolCalls.length, 0)
  assert.match(result.malformedReason || '', /Malformed DeepSeek DSML/)
})

test('deepseek dsml reports malformed native-looking output explicitly', () => {
  const result = deepseekDsmlProtocol.parse(
    '<｜DSML｜tool_calls><｜DSML｜invoke name="default_api:read_file"><｜DSML｜parameter name="filePath" string="true">/tmp/a</｜DSML｜invoke>',
    { tools, protocol: 'deepseek_dsml' },
  )

  assert.equal(result.protocol, 'deepseek_dsml')
  assert.equal(result.toolCalls.length, 0)
  assert.match(result.malformedReason || '', /Malformed DeepSeek DSML/)
})

test('managed xml ignores fenced tool examples', () => {
  const result = managedXmlProtocol.parse(
    '```xml\n<|CHAT2API|tool_calls><|CHAT2API|invoke name="default_api:read_file"><|CHAT2API|parameter name="filePath">fake</|CHAT2API|parameter></|CHAT2API|invoke></|CHAT2API|tool_calls>\n```',
    { tools, protocol: 'managed_xml' },
  )

  assert.equal(result.toolCalls.length, 0)
})

test('unknown tool name is rejected', () => {
  const result = managedBracketProtocol.parse(
    '[function_calls][call:missing_tool]{"x":1}[/call][/function_calls]',
    { tools, protocol: 'managed_bracket' },
  )

  assert.equal(result.toolCalls.length, 0)
  assert.deepEqual(result.invalidToolNames, ['missing_tool'])
})

test('managed XML parser rejects undeclared tool names and records invalid names', () => {
  const result = managedXmlProtocol.parse(
    '<|CHAT2API|tool_calls><|CHAT2API|invoke name="missing_tool">{}</|CHAT2API|invoke></|CHAT2API|tool_calls>',
    { tools, protocol: 'managed_xml' },
  )

  assert.equal(result.toolCalls.length, 0)
  assert.deepEqual(result.invalidToolNames, ['missing_tool'])
})

test('anthropic adapter parses antml function calls', () => {
  const result = anthropicToolUseProtocol.parse(
    '<antml:function_calls><antml:invoke name="default_api:read_file"><antml:parameters>{"filePath":"/tmp/a"}</antml:parameters></antml:invoke></antml:function_calls>',
    { tools, protocol: 'anthropic_tool_use' },
  )

  assert.equal(result.toolCalls.length, 1)
})

test('codex responses adapter parses response item function call', () => {
  const result = codexResponsesProtocol.parse(
    JSON.stringify({
      type: 'function_call',
      call_id: 'call_1',
      name: 'default_api:read_file',
      arguments: '{"filePath":"/tmp/a"}',
    }),
    { tools, protocol: 'codex_responses' },
  )

  assert.equal(result.toolCalls.length, 1)
  assert.equal(result.toolCalls[0].id, 'call_1')
})
