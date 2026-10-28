import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildTaskBody, defaultModelFor, describeChoice, describeMeta, isValidChoice, normaliseProviderChoices,
  parseStoredChoice, pickInitialChoice, readStoredChoice, writeStoredChoice, STORAGE_KEY,
} from './modelChoice.mjs';

const NEW_FORMAT = [
  { id: 'structured', label: 'Structured parser', available: true, reason: null, models: [], default_model: null, supports: ['inspect cabinet', 'goto <place>'] },
  { id: 'qwen', label: 'Qwen', available: false, reason: 'start the backend with --qwen-url', models: [], default_model: null, supports: ['inspect cabinet'] },
  { id: 'openai_chatgpt_plan', label: 'ChatGPT', available: true, reason: null, models: ['gpt-6-luna', 'gpt-6-sol'], default_model: 'gpt-6-sol', supports: ['inspect cabinet'] },
];

test('normalises the new format and keeps reasons, models and supports', () => {
  const [structured, qwen, chatgpt] = normaliseProviderChoices(NEW_FORMAT);
  assert.equal(structured.available, true);
  assert.deepEqual(structured.supports, ['inspect cabinet', 'goto <place>']);
  assert.equal(qwen.available, false);
  assert.equal(qwen.reason, 'start the backend with --qwen-url');
  assert.deepEqual(chatgpt.models, ['gpt-6-luna', 'gpt-6-sol']);
  assert.equal(chatgpt.default_model, 'gpt-6-sol');
});

test('treats an old {id, label} snapshot as available with no models', () => {
  const old = normaliseProviderChoices([{ id: 'structured', label: 'Structured parser' }, { id: 'qwen', label: 'Qwen' }]);
  assert.deepEqual(old.map((c) => [c.id, c.available, c.models, c.reason, c.supports]), [
    ['structured', true, [], null, []],
    ['qwen', true, [], null, []],
  ]);
});

test('drops malformed entries and non-array input instead of throwing', () => {
  assert.deepEqual(normaliseProviderChoices(undefined), []);
  assert.deepEqual(normaliseProviderChoices('qwen'), []);
  const cleaned = normaliseProviderChoices([null, 3, { label: 'no id' }, { id: 'qwen', label: 'Qwen', models: ['a', 7, ''] }]);
  assert.deepEqual(cleaned.map((c) => c.id), ['qwen']);
  assert.deepEqual(cleaned[0].models, ['a']);
});

test('an empty label falls back to the id', () => {
  assert.equal(normaliseProviderChoices([{ id: 'qwen', label: '' }])[0].label, 'qwen');
});

test('defaultModelFor prefers default_model, then the first model, then none', () => {
  const [, , chatgpt] = normaliseProviderChoices(NEW_FORMAT);
  assert.equal(defaultModelFor(chatgpt), 'gpt-6-sol');
  assert.equal(defaultModelFor({ ...chatgpt, default_model: 'gone' }), 'gpt-6-luna');
  assert.equal(defaultModelFor({ ...chatgpt, default_model: null, models: [] }), null);
  assert.equal(defaultModelFor(undefined), null);
});

test('isValidChoice rejects unavailable providers, unknown models and a model for structured', () => {
  const choices = normaliseProviderChoices(NEW_FORMAT);
  assert.equal(isValidChoice(choices, { provider: 'structured', model: null }), true);
  assert.equal(isValidChoice(choices, { provider: 'structured', model: 'x' }), false);
  assert.equal(isValidChoice(choices, { provider: 'qwen', model: null }), false);
  assert.equal(isValidChoice(choices, { provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' }), true);
  assert.equal(isValidChoice(choices, { provider: 'openai_chatgpt_plan', model: 'retired' }), false);
  assert.equal(isValidChoice(choices, { provider: 'openai_chatgpt_plan', model: null }), false);
  assert.equal(isValidChoice(choices, { provider: 'nope', model: null }), false);
  assert.equal(isValidChoice(choices, null), false);
});

test('pickInitialChoice keeps a valid stored choice', () => {
  const choices = normaliseProviderChoices(NEW_FORMAT);
  assert.deepEqual(pickInitialChoice(choices, 'structured', { provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' }),
    { provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' });
});

test('pickInitialChoice ignores a stored choice that is no longer valid', () => {
  const choices = normaliseProviderChoices(NEW_FORMAT);
  assert.deepEqual(pickInitialChoice(choices, 'structured', { provider: 'qwen', model: null }), { provider: 'structured', model: null });
  assert.deepEqual(pickInitialChoice(choices, 'openai_chatgpt_plan', { provider: 'openai_chatgpt_plan', model: 'retired' }),
    { provider: 'openai_chatgpt_plan', model: 'gpt-6-sol' });
});

test('pickInitialChoice falls back to the default provider and its default model', () => {
  const choices = normaliseProviderChoices(NEW_FORMAT);
  assert.deepEqual(pickInitialChoice(choices, 'openai_chatgpt_plan', null), { provider: 'openai_chatgpt_plan', model: 'gpt-6-sol' });
});

test('pickInitialChoice skips an unavailable default and takes the first available provider', () => {
  const choices = normaliseProviderChoices(NEW_FORMAT);
  assert.deepEqual(pickInitialChoice(choices, 'qwen', undefined), { provider: 'structured', model: null });
});

test('pickInitialChoice returns null when nothing is available', () => {
  const choices = normaliseProviderChoices([{ id: 'qwen', label: 'Qwen', available: false, reason: 'no url' }]);
  assert.equal(pickInitialChoice(choices, 'qwen', null), null);
  assert.equal(pickInitialChoice([], 'structured', null), null);
});

test('buildTaskBody omits model for structured and includes it for a model provider', () => {
  assert.deepEqual(buildTaskBody('inspect cabinet', { provider: 'structured', model: null }), { text: 'inspect cabinet', priority: 0, provider: 'structured' });
  assert.deepEqual(buildTaskBody('goto entrance', { provider: 'structured', model: null }, 2), { text: 'goto entrance', priority: 2, provider: 'structured' });
  assert.deepEqual(buildTaskBody('inspect cabinet', { provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' }),
    { text: 'inspect cabinet', priority: 0, provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' });
});

test('buildTaskBody omits model when a model provider has none chosen', () => {
  const body = buildTaskBody('inspect cabinet', { provider: 'openai_chatgpt_plan', model: null });
  assert.equal('model' in body, false);
});

test('stored choice round-trips through a storage-like object under the expected key', () => {
  const data = new Map();
  const storage = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v) };
  assert.equal(writeStoredChoice(storage, { provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' }), true);
  assert.equal(data.get('dream-model-choice'), JSON.stringify({ provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' }));
  assert.equal(STORAGE_KEY, 'dream-model-choice');
  assert.deepEqual(readStoredChoice(storage), { provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' });
});

test('storage failures are swallowed', () => {
  const throwing = {
    getItem() { throw new Error('blocked'); },
    setItem() { throw new Error('quota'); },
  };
  assert.equal(readStoredChoice(throwing), null);
  assert.equal(writeStoredChoice(throwing, { provider: 'structured', model: null }), false);
  assert.equal(readStoredChoice(null), null);
  assert.equal(writeStoredChoice(null, { provider: 'structured', model: null }), true);
});

test('malformed stored values are ignored', () => {
  assert.equal(parseStoredChoice('not json'), null);
  assert.equal(parseStoredChoice('null'), null);
  assert.equal(parseStoredChoice('{"provider":3}'), null);
  assert.equal(parseStoredChoice('{"provider":"qwen","model":5}'), null);
  assert.deepEqual(parseStoredChoice('{"provider":"structured"}'), { provider: 'structured', model: null });
  assert.equal(parseStoredChoice(''), null);
});

test('describeChoice and describeMeta give the trigger and thread labels', () => {
  const choices = normaliseProviderChoices(NEW_FORMAT);
  assert.equal(describeChoice(choices, { provider: 'structured', model: null }), 'Structured parser');
  assert.equal(describeChoice(choices, { provider: 'openai_chatgpt_plan', model: 'gpt-6-luna' }), 'ChatGPT · gpt-6-luna');
  assert.equal(describeChoice(choices, { provider: 'gone', model: null }), 'gone');
  assert.equal(describeChoice(choices, null), 'No provider available');
  assert.equal(describeMeta(choices, { provider: 'openai_chatgpt_plan', model: 'gpt-6-sol' }), 'ChatGPT · gpt-6-sol');
  assert.equal(describeMeta(choices, { provider: 'structured', model: null }), 'Structured parser');
  assert.equal(describeMeta(choices, undefined), null);
  assert.equal(describeMeta(choices, {}), null);
});
