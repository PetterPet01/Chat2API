import test from 'node:test'
import assert from 'node:assert/strict'
import { getProviderToolProfile } from '../../src/main/proxy/toolCalling/providerProfiles.ts'

const calls = [
  { id: 'call_1', name: 'default_api:read_file', arguments: '{"filePath":"/tmp/a"}' },
]

test('deepseek uses native DSML while other first-version providers keep managed XML', () => {
  const deepseek = getProviderToolProfile('deepseek')
  assert.equal(deepseek.managedSupport, true)
  assert.equal(deepseek.supportsNativeTools, false)
  assert.equal(deepseek.preferredManagedProtocol, 'deepseek_dsml')

  for (const providerId of ['kimi', 'glm', 'qwen']) {
    const profile = getProviderToolProfile(providerId)

    assert.equal(profile.managedSupport, true)
    assert.equal(profile.supportsNativeTools, false)
    assert.equal(profile.preferredManagedProtocol, 'managed_xml')
  }
})

test('priority providers format tool history with provider-specific protocols', () => {
  const deepseek = getProviderToolProfile('deepseek')
  assert.equal(
    deepseek.formatAssistantToolCalls(calls),
    '<｜DSML｜tool_calls><｜DSML｜invoke name="default_api:read_file"><｜DSML｜parameter name="filePath" string="true">/tmp/a</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>',
  )
  assert.equal(
    deepseek.formatToolResult({ toolCallId: 'call_1', content: 'file body' }),
    '<tool_result>file body</tool_result>',
  )

  for (const providerId of ['kimi', 'glm', 'qwen']) {
    const profile = getProviderToolProfile(providerId)

    assert.equal(
      profile.formatAssistantToolCalls(calls),
      '<|CHAT2API|tool_calls><|CHAT2API|invoke name="default_api:read_file"><|CHAT2API|parameter name="filePath"><![CDATA[/tmp/a]]></|CHAT2API|parameter></|CHAT2API|invoke></|CHAT2API|tool_calls>',
    )
    assert.equal(
      profile.formatToolResult({ toolCallId: 'call_1', content: 'file body' }),
      '<|CHAT2API|tool_result tool_call_id="call_1"><![CDATA[file body]]></|CHAT2API|tool_result>',
    )
  }
})
