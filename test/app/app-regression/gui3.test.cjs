const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {
  readSource,
  assertIncludes,
  assertMatches,
  normalizeWhitespace,
} = require('./helpers/source.cjs');

const COLLECTION_MODELS = 'src/app/src/features/collections/collectionModels.ts';
const CHAT_VIEW = 'src/app/src/components/ChatView.tsx';
const CHAT_ATTACHMENTS = path.join(
  __dirname, '..', '..', '..', 'src', 'app', 'src', 'features', 'chatAttachments', 'fileAttachments.ts',
);
const CHAT_QUEUE = path.join(
  __dirname, '..', '..', '..', 'src', 'app', 'src', 'features', 'chatQueue.ts',
);

// Transpile-on-require so the pure feature helpers can be unit-tested here
// (same loader pattern as the legacy fileAttachments suite).
function loadTsModule(absPath) {
  let ts = null;
  try { ts = require(path.join(__dirname, '..', '..', '..', 'src', 'app', 'node_modules', 'typescript')); }
  catch (_) {
    try { ts = require('typescript'); } catch (_2) { return null; }
  }
  if (!require.extensions['.ts']) {
    require.extensions['.ts'] = function loadTypeScript(module, filename) {
      const source = fs.readFileSync(filename, 'utf8');
      const output = ts.transpileModule(source, {
        compilerOptions: {
          esModuleInterop: true, module: ts.ModuleKind.CommonJS,
          moduleResolution: ts.ModuleResolutionKind.NodeJs, target: ts.ScriptTarget.ES2020,
        },
        fileName: filename,
      }).outputText;
      module._compile(output, filename);
    };
  }
  return require(absPath);
}
function loadChatAttachments() { return loadTsModule(CHAT_ATTACHMENTS); }
function loadChatQueue() { return loadTsModule(CHAT_QUEUE); }
const MODEL_MANAGER = 'src/app/src/components/ModelManager.tsx';
const OMNI_TOOLS = 'src/app/src/tools/omniTools.ts';
const TOOL_DEFINITIONS = 'src/app/src/tools/toolDefinitions.json';

const tests = [
  {
    name: 'collection helpers preserve component order and collection identity',
    run() {
      const source = normalizeWhitespace(readSource(COLLECTION_MODELS));
      assertMatches(
        source,
        /const candidates = \[[\s\S]*?model as any\)\.components[\s\S]*?model as any\)\.component_models[\s\S]*?model as any\)\.recipe_options\?\.components/,
        'Collection components should accept the GUI3 model shapes.',
      );
      assertIncludes(
        source,
        'Array.from(new Set(raw.filter',
        'Collection components should be deduplicated without changing their declared order.',
      );
      assertMatches(
        source,
        /isCollectionModel[\s\S]*?isCollectionRecipe\(\(model as any\)\.recipe\)[\s\S]*?getCollectionComponents\(model\)\.length > 0/,
        'A collection model should require the collection recipe and at least one component.',
      );
    },
  },
  {
    name: 'collection loading state requires every concrete component',
    run() {
      const source = normalizeWhitespace(readSource(COLLECTION_MODELS));
      assertMatches(
        source,
        /isCollectionFullyDownloaded[\s\S]*?components\.every[\s\S]*?downloaded === true/,
        'A collection should be downloaded only when every component is downloaded.',
      );
      assertMatches(
        source,
        /isCollectionFullyLoaded[\s\S]*?components\.every[\s\S]*?loaded\.has\(component\.toLowerCase\(\)\)/,
        'A collection should be loaded only when every component is loaded.',
      );
    },
  },
  {
    name: 'ModelScope results are validated through bounded variant lookups',
    run() {
      const source = normalizeWhitespace(readSource(MODEL_MANAGER));
      assertIncludes(source, 'searchModelScope(q, ac.signal)', 'GUI3 should use the ModelScope search API.');
      assertIncludes(
        source,
        "loadRemoteVariants('modelscope', candidate.id, ac.signal)",
        'ModelScope results should be checked for usable variants before display.',
      );
      assertMatches(
        source,
        /Promise\.all\(Array\.from\(\{ length: Math\.min\(REMOTE_VARIANT_CONCURRENCY, candidates\.length\)/,
        'Remote variant validation should remain concurrency-bounded.',
      );
    },
  },
  {
    name: 'GUI3 Omni definitions keep planner guidance and required media tools',
    run() {
      const definitions = JSON.parse(readSource(TOOL_DEFINITIONS));
      const names = definitions.tools.map((tool) => tool.function.name).sort();
      assert.deepEqual(names, ['edit_image', 'generate_image', 'text_to_speech']);
      assertIncludes(definitions.system_prompt, '{tool_list}', 'The Omni planner prompt should reserve the tool list slot.');
      assertIncludes(definitions.system_prompt, '{tool_guidance}', 'The Omni planner prompt should reserve the guidance slot.');

      const source = normalizeWhitespace(readSource(OMNI_TOOLS));
      assertIncludes(source, 'DEFAULT_OMNI_SYSTEM_PROMPT_TEMPLATE', 'GUI3 should keep a shared Omni prompt template.');
      assertIncludes(source, 'renderOmniSystemPrompt', 'GUI3 should render the planner prompt from available tools.');
      assertIncludes(source, 'resolveExplicitImageSize', 'GUI3 should support explicit image dimensions.');
    },
  },
  {
    name: 'GUI3 ModelManager loads collection components while keeping the collection virtual',
    run() {
      const source = normalizeWhitespace(readSource(MODEL_MANAGER));
      assertIncludes(
        source,
        'const components = info && isCollectionModel(info) ? getCollectionComponents(info) : []',
        'ModelManager should identify collection components before loading a model.',
      );
      assertMatches(
        source,
        /if \(components\.length > 0\)[\s\S]*?for \(const componentName of components\)[\s\S]*?loadModelRuntime\(componentInfo \|\| componentName, visited\)/,
        'Loading a collection should recursively load its concrete components.',
      );
      assertIncludes(
        source,
        'selected virtual model in the UI',
        'The collection should remain a virtual selection after its components load.',
      );
    },
  },
  {
    name: 'chat attachments mint stable ids and stay readable for pre-id history',
    run() {
      const fa = loadChatAttachments();
      if (!fa) return { skip: true, reason: "typescript not installed - run 'npm ci' in src/app first" };
      const first = fa.createAttachmentId();
      const second = fa.createAttachmentId();
      assert.notEqual(first, second);
      assert.match(first, /^file-\d+$/);
      const legacy = { filename: 'old.md', language: 'markdown', content: '# hi', size: 4 };
      assert.ok(fa.wrapFileForPrompt(legacy).includes('Attached file: old.md'));
      assert.ok(fa.composePromptWithFiles('look', [legacy]).startsWith('look'));
      const nested = fa.wrapFileForPrompt({ ...legacy, content: '```nested```' });
      assert.ok(nested.includes('````markdown'), 'fence must outgrow backtick runs in content');
    },
  },
  {
    name: 'chat queue helpers cap at ten, mint unique ids, and summarize drafts',
    run() {
      const q = loadChatQueue();
      if (!q) return { skip: true, reason: "typescript not installed - run 'npm ci' in src/app first" };
      assert.equal(q.MAX_QUEUED_MESSAGES, 10);
      const first = q.createQueuedMessageId();
      const second = q.createQueuedMessageId();
      assert.match(first, /^queued-/);
      assert.notEqual(first, second, 'ids minted in the same millisecond must stay unique');

      let queue = [];
      for (let i = 0; i < q.MAX_QUEUED_MESSAGES; i += 1) {
        const result = q.withQueueCap(queue, { id: `m${i}`, text: `follow-up ${i}` });
        assert.equal(result.dropped, false);
        queue = result.kept;
      }
      const overflow = q.withQueueCap(queue, { id: 'overflow', text: 'one too many' });
      assert.equal(overflow.dropped, true, 'the 11th message must be rejected');
      assert.equal(overflow.kept, queue, 'a rejection must return the original queue untouched');

      assert.equal(q.summarizeQueuedItem({ id: 'a', text: 'plain question' }), 'plain question');
      assert.ok(q.summarizeQueuedItem({ id: 'a', text: 'x'.repeat(80) }).endsWith('…'));
      assert.equal(
        q.summarizeQueuedItem({ id: 'a', text: '', files: [{ filename: 'alpha.txt', language: 'text', content: '', size: 1 }] }),
        'File: alpha.txt',
      );
      assert.equal(q.summarizeQueuedItem({ id: 'a', text: '   ' }), 'Empty message');
    },
  },
  {
    name: 'ChatView keys attachment chips on stable ids and explains rejected drops',
    run() {
      const source = readSource(CHAT_VIEW);
      assertIncludes(source, 'key={file.id ?? file.filename}', 'Composer chips must key on the attachment id.');
      assertIncludes(
        source,
        'key={file.id ?? `${file.filename}-${i}`}',
        'History chips must fall back for storage saved before ids existed.',
      );
      assert.equal(source.split('id: createAttachmentId(),').length - 1, 2, 'Both ingest paths must mint attachment ids.');
      assertIncludes(source, 'prev.filter(f => f.id !== id)', 'Removal must target the attachment id, not a shifted index.');
      assertIncludes(source, 'not attachable in this mode', 'Modes without chat completions must explain rejected documents.');
    },
  },
];

module.exports = { tests };
