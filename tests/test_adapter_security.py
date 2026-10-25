import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from table_lora import adapter_security
from table_lora.adapter_security import adapter_selection, load_selection, load_safe_weights, require_safe_options


class AdapterSecurityTests(unittest.TestCase):
    def test_default_never_discovers_siblings(self):
        with patch("os.listdir", side_effect=AssertionError("Directory discovery is forbidden")):
            self.assertEqual(adapter_selection(), {"default": None})
            self.assertEqual(adapter_selection("custom", "chosen"), {"custom": "chosen"})

    def test_explicit_multiple_adapters(self):
        selection = {"default": None, "table": "table", "other": "nested/other"}
        self.assertEqual(adapter_selection(adapter_subfolders=selection), selection)

    def test_invalid_names_and_subfolders(self):
        for path in ("../outside", "/tmp/weights", "a/../../b", "a\\b", "C:/weights", "a//b", "%2e%2e", "a\x00"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                adapter_selection(subfolder=path)
        with self.assertRaises(ValueError):
            adapter_selection(adapter_subfolders={"other": "other"})
        with self.assertRaises(ValueError):
            adapter_selection(subfolder="first", adapter_subfolders={"default": None})

    def test_unsafe_options_are_rejected(self):
        for options in ({"use_safetensors": False}, {"use_safetensors": None}, {"trust_remote_code": True}, {"adapters": {"hidden": "model"}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                require_safe_options(options)

    def test_direct_loading_selection(self):
        self.assertEqual(load_selection("renamed"), {"renamed": None})
        self.assertEqual(load_selection(["table"]), {"table": "table"})
        self.assertEqual(load_selection(["default", "table"]), {"default": None, "table": "table"})
        for names in ([], ["default", "default"], ["../bad"], 42):
            with self.subTest(names=names), self.assertRaises(ValueError):
                load_selection(names)
        with self.assertRaises(ValueError):
            load_selection(["default"], adapter_subfolders={"default": None, "hidden": "hidden"})

    def test_pickle_only_adapter_is_never_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "adapter_model.bin").write_bytes(b"not a trusted pickle")
            decoder = types.ModuleType("safetensors.torch")
            decoder.load = lambda data: self.fail("Missing safetensors must fail before decoding")
            with patch.dict(sys.modules, {"safetensors.torch": decoder}), self.assertRaises(ValueError):
                load_safe_weights(directory)

    def test_only_selected_safetensors_bytes_reach_decoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chosen").mkdir()
            (root / "unrequested").mkdir()
            (root / "chosen" / "adapter_model.safetensors").write_bytes(b"selected safe bytes")
            (root / "unrequested" / "adapter_model.bin").write_bytes(b"must never load")
            decoder = types.ModuleType("safetensors.torch")
            received = []
            decoder.load = lambda data: received.append(data) or {"weight": "tensor"}
            with patch.dict(sys.modules, {"safetensors.torch": decoder}):
                self.assertEqual(load_safe_weights(directory, "chosen"), {"weight": "tensor"})
            self.assertEqual(received, [b"selected safe bytes"])

    @unittest.skipUnless(hasattr(os, "symlink"), "Symlink support required")
    def test_symlink_outside_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            root.mkdir()
            outside = Path(directory) / "outside.safetensors"
            outside.write_bytes(b"outside")
            try:
                (root / "adapter_model.safetensors").symlink_to(outside)
            except OSError:
                self.skipTest("Symlink creation is not permitted")
            with self.assertRaises(ValueError):
                load_safe_weights(root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO support required")
    def test_non_regular_weights_are_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            os.mkfifo(Path(directory) / "adapter_model.safetensors")
            with self.assertRaises(ValueError):
                load_safe_weights(directory)

    def test_weights_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "adapter_model.safetensors").write_bytes(b"12345")
            with patch.object(adapter_security, "MAX_WEIGHTS_BYTES", 4), self.assertRaises(ValueError):
                load_safe_weights(directory)

    @unittest.skipUnless(hasattr(os, "symlink"), "Directory-race regression requires symlinks")
    def test_parent_directory_swap_cannot_read_outside_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            chosen = root / "chosen"
            chosen.mkdir(parents=True)
            outside = Path(directory) / "outside"
            outside.mkdir()
            weight_path = chosen / "adapter_model.safetensors"
            weight_path.write_bytes(b"selected bytes")
            (outside / "adapter_model.safetensors").write_bytes(b"outside bytes")
            real_open = os.open
            swapped = []
            def replace_directory(path, flags, *args, **kwargs):
                if not swapped:
                    chosen.rename(root / "original_chosen")
                    try:
                        chosen.symlink_to(outside, target_is_directory=True)
                    except OSError:
                        if os.name == "nt":
                            self.skipTest("Symlink creation is not permitted")
                        raise
                    swapped.append(True)
                return real_open(path, flags, *args, **kwargs)
            decoder = types.ModuleType("safetensors.torch")
            received = []
            decoder.load = lambda data: received.append(data) or {}
            with patch.dict(sys.modules, {"safetensors.torch": decoder}):
                with patch.object(adapter_security.os, "open", side_effect=replace_directory) as hooked_open:
                    with patch.object(adapter_security.os, "supports_dir_fd", os.supports_dir_fd | {hooked_open}):
                        try:
                            load_safe_weights(root, "chosen")
                        except (OSError, ValueError):
                            pass
            self.assertNotIn(b"outside bytes", received)

    def test_hub_requests_only_selected_safetensors(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "adapter_model.safetensors"
            weights.write_bytes(b"safe bytes")
            calls = []
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = lambda *args, **kwargs: calls.append((args, kwargs)) or str(weights)
            decoder = types.ModuleType("safetensors.torch")
            decoder.load = lambda data: {"weight": "tensor"}
            with patch.dict(sys.modules, {"huggingface_hub": hub, "safetensors.torch": decoder}):
                load_safe_weights("test-owner/test-model", "chosen", revision="reviewed-commit", local_files_only=True)
            self.assertEqual(calls, [(("test-owner/test-model",), {"filename": "adapter_model.safetensors", "subfolder": "chosen", "revision": "reviewed-commit", "local_files_only": True})])

    def test_missing_hub_safe_weights_never_requests_bin(self):
        requests = []
        hub = types.ModuleType("huggingface_hub")
        def missing(*args, **kwargs):
            requests.append(kwargs["filename"])
            raise FileNotFoundError("No safetensors weights")
        hub.hf_hub_download = missing
        with patch.dict(sys.modules, {"huggingface_hub": hub}), self.assertRaises(FileNotFoundError):
            load_safe_weights("test-owner/test-model")
        self.assertEqual(requests, ["adapter_model.safetensors"])

    def test_hub_cache_symlink_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blob = root / "blob"
            blob.write_bytes(b"safe cached bytes")
            link = root / "adapter_model.safetensors"
            try:
                link.symlink_to(blob)
            except OSError:
                self.skipTest("Symlink creation is not permitted")
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = lambda *args, **kwargs: str(link)
            decoder = types.ModuleType("safetensors.torch")
            decoder.load = lambda data: {"cached": data}
            with patch.dict(sys.modules, {"huggingface_hub": hub, "safetensors.torch": decoder}):
                self.assertEqual(load_safe_weights("test-owner/test-model"), {"cached": b"safe cached bytes"})


@unittest.skipUnless(importlib.util.find_spec("safetensors") and importlib.util.find_spec("numpy"), "Real codec tests require safetensors and numpy")
class SafetensorsCodecTests(unittest.TestCase):
    def setUp(self):
        import safetensors.numpy
        self.codec = safetensors.numpy
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        decoder = types.ModuleType("safetensors.torch")
        decoder.load = self.codec.load
        self.modules = patch.dict(sys.modules, {"safetensors.torch": decoder})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_real_safe_weights_decode(self):
        import numpy as np
        (self.root / "adapter_model.safetensors").write_bytes(self.codec.save({"weight": np.array([1.0, 2.0], dtype=np.float32)}))
        loaded = load_safe_weights(self.root)
        np.testing.assert_array_equal(loaded["weight"], np.array([1.0, 2.0], dtype=np.float32))

    def test_renamed_pickle_has_no_side_effects(self):
        sentinel = self.root / "executed.txt"
        class Payload:
            def __reduce__(self):
                return eval, ("__import__('pathlib').Path(" + repr(str(sentinel)) + ").write_text('executed')",)
        (self.root / "adapter_model.safetensors").write_bytes(pickle.dumps(Payload()))
        with self.assertRaises(Exception):
            load_safe_weights(self.root)
        self.assertFalse(sentinel.exists())

    def test_corrupt_safetensors_does_not_fall_back_to_bin(self):
        (self.root / "adapter_model.safetensors").write_bytes(b"invalid format")
        (self.root / "adapter_model.bin").write_bytes(b"must never deserialize")
        with patch.object(pickle, "load", side_effect=AssertionError("Pickle loader called")) as unsafe_load:
            with self.assertRaises(Exception):
                load_safe_weights(self.root)
            unsafe_load.assert_not_called()


class LoadingMethodTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model_dir = Path(self.temporary.name)
        self.config_calls = []
        self.state_calls = []
        self.decode_calls = []
        config_calls = self.config_calls
        class Config:
            task_type = "test"
            is_prompt_learning = False
            inference_mode = True
            @staticmethod
            def _get_peft_type(model_id, subfolder=None, **kwargs):
                config_calls.append(subfolder)
                folder = Path(model_id) / subfolder if subfolder else Path(model_id)
                return json.loads((folder / "adapter_config.json").read_text())["peft_type"]
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                return cls()
        self.config_class = Config
        mapping = types.ModuleType("peft.mapping")
        mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {}
        mapping.PEFT_TYPE_TO_CONFIG_MAPPING = {"LORA": Config}
        peft = types.ModuleType("peft")
        peft.__path__ = []
        peft.mapping = mapping
        decoder = types.ModuleType("safetensors.torch")
        decoder.load = lambda data: self.decode_calls.append(data) or {"weight": "tensor"}
        modules = patch.dict(sys.modules, {"peft": peft, "peft.mapping": mapping, "safetensors.torch": decoder})
        modules.start()
        self.addCleanup(modules.stop)
        source = ast.parse((ROOT / "table_lora" / "table_lora.py").read_text(encoding="utf-8"))
        original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PeftModel_new")
        methods = [copy.deepcopy(node) for node in original.body if isinstance(node, ast.FunctionDef) and node.name in ("from_pretrained", "load_adapter")]
        loader = ast.ClassDef(name="Loader", bases=[], keywords=[], body=methods, decorator_list=[])
        tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), loader], type_ignores=[])
        self.namespace = {"adapter_selection": adapter_selection, "load_selection": load_selection,
                          "load_safe_weights": load_safe_weights, "require_safe_options": require_safe_options,
                          "PeftConfig": Config, "XLoraConfig": type("XLoraConfig", (Config,), {}),
                          "XLoraModel": type("XLoraModel", (), {}), "os": os,
                          "infer_device": lambda: "cpu", "set_peft_model_state_dict": lambda model, weights, **kwargs: self.state_calls.append(kwargs["adapter_name"])}
        exec(compile(ast.fix_missing_locations(tree), "<actual-loading-methods>", "exec"), self.namespace)
        self.loader = self.namespace["Loader"]
        def initialize(model, base, config, adapter_name, **kwargs):
            model.peft_config = {adapter_name: config}
            model.base_model = types.SimpleNamespace()
        self.loader.__init__ = initialize
        self.loader._split_kwargs = lambda model, kwargs: (dict(kwargs), {})
        self.loader.add_adapter = lambda model, name, config: model.peft_config.update({name: config})
        self.loader.eval = lambda model: model

    def checkpoint(self, subfolder=None, safe=True):
        folder = self.model_dir / subfolder if subfolder else self.model_dir
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (folder / ("adapter_model.safetensors" if safe else "adapter_model.bin")).write_bytes(b"safe weights" if safe else b"pickle must not load")

    def load(self, **kwargs):
        return self.loader.from_pretrained(types.SimpleNamespace(), str(self.model_dir), **kwargs)

    def test_default_load_ignores_unrequested_pickle_subfolder(self):
        self.checkpoint()
        self.checkpoint("unrequested", safe=False)
        with patch("os.listdir", side_effect=AssertionError("No automatic discovery")):
            self.load()
        self.assertEqual(self.config_calls, [None])
        self.assertEqual(self.state_calls, ["default"])
        self.assertEqual(self.decode_calls, [b"safe weights"])

    def test_custom_name_and_explicit_subfolder(self):
        self.checkpoint("chosen")
        self.load(adapter_name="renamed", subfolder="chosen")
        self.assertEqual(self.config_calls, ["chosen"])
        self.assertEqual(self.state_calls, ["renamed"])

    def test_explicit_multiple_adapters_only(self):
        self.checkpoint()
        self.checkpoint("table")
        self.checkpoint("unrequested", safe=False)
        self.load(adapter_subfolders={"table": "table", "default": None})
        self.assertEqual(self.state_calls, ["default", "table"])
        self.assertEqual(self.config_calls, [None, "table", "table"])

    def test_bad_requested_config_is_not_silently_skipped(self):
        self.checkpoint("unrequested")
        with self.assertRaises(FileNotFoundError):
            self.load()
        self.assertEqual(self.state_calls, [])

    def test_pickle_only_root_is_rejected(self):
        self.checkpoint(safe=False)
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(self.state_calls, [])
        self.assertEqual(self.decode_calls, [])

    def test_direct_load_adapter_cannot_bypass_safe_format(self):
        self.checkpoint(safe=False)
        model = self.loader(types.SimpleNamespace(), self.config_class(), "default")
        with self.assertRaises(ValueError):
            model.load_adapter(str(self.model_dir), "default")
        with self.assertRaises(ValueError):
            model.load_adapter(str(self.model_dir), "default", use_safetensors=False)
        self.assertEqual(self.state_calls, [])

    def test_nested_xlora_is_rejected(self):
        self.checkpoint()
        with self.assertRaises(ValueError):
            self.load(config=self.namespace["XLoraConfig"]())
        with self.assertRaises(ValueError):
            self.load(adapters={"unrequested": "somewhere"})
        self.assertEqual(self.state_calls, [])

    def test_no_generic_pickle_loader_remains(self):
        source = (ROOT / "table_lora" / "table_lora.py").read_text(encoding="utf-8")
        self.assertNotIn("load_peft_weights", source)
        self.assertNotIn("HfFileSystem", source)
        self.assertNotIn("os.listdir(model_id)", source)


@unittest.skipUnless(os.environ.get("TABLELORA_TEST_REAL_PEFT") == "1", "Explicitly enable the isolated real PEFT/PyTorch smoke tests")
class RealFrameworkTests(unittest.TestCase):
    def run_case(self, case):
        script = '''import os
import pickle
from pathlib import Path
import shutil
import sys
import tempfile
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict
from table_lora.table_lora import load_table_lora

configuration = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16, n_embd=8, n_layer=1, n_head=1)
load_table_lora()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    task_type = None if sys.argv[1] == "generic" else TaskType.CAUSAL_LM
    reference = get_peft_model(GPT2LMHeadModel(configuration), LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type=task_type))
    reference.save_pretrained(root, safe_serialization=True)
    expected = {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(reference).items()}
    chosen = root / "chosen"
    chosen.mkdir()
    shutil.copyfile(root / "adapter_config.json", chosen / "adapter_config.json")
    shutil.copyfile(root / "adapter_model.safetensors", chosen / "adapter_model.safetensors")
    sentinel = root / "executed.txt"
    class Payload:
        def __reduce__(self):
            return eval, ("__import__('pathlib').Path(" + repr(str(sentinel)) + ").write_text('executed')",)
    unrequested = root / "unrequested"
    unrequested.mkdir()
    shutil.copyfile(root / "adapter_config.json", unrequested / "adapter_config.json")
    (unrequested / "adapter_model.bin").write_bytes(pickle.dumps(Payload()))
    def reject_pickle(*args, **kwargs):
        raise AssertionError("torch.load must not be called")
    torch.load = reject_pickle
    if sys.argv[1] in ("default", "generic"):
        model = PeftModel.from_pretrained(GPT2LMHeadModel(configuration), root)
        assert list(model.peft_config) == ["default"]
        actual = get_peft_model_state_dict(model)
        assert set(actual) == set(expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])
    elif sys.argv[1] == "multiple":
        model = PeftModel.from_pretrained(GPT2LMHeadModel(configuration), root, adapter_subfolders={"default": None, "second": "chosen"})
        assert set(model.peft_config) == {"default", "second"}
    elif sys.argv[1] == "subfolder":
        model = PeftModel.from_pretrained(GPT2LMHeadModel(configuration), root, adapter_name="renamed", subfolder="chosen")
        assert list(model.peft_config) == ["renamed"]
        actual = get_peft_model_state_dict(model, adapter_name="renamed")
        assert set(actual) == set(expected)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])
        model.load_adapter(root, "third", subfolder="chosen")
        assert set(model.peft_config) == {"renamed", "third"}
    else:
        (root / "adapter_model.safetensors").unlink()
        (root / "adapter_model.bin").write_bytes(pickle.dumps(Payload()))
        try:
            PeftModel.from_pretrained(GPT2LMHeadModel(configuration), root)
        except ValueError:
            pass
        else:
            raise AssertionError("Pickle-only checkpoint was accepted")
    assert not sentinel.exists()
print("real-framework case passed")
'''
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        result = subprocess.run([sys.executable, "-B", "-c", script, case], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_default_loading(self):
        self.run_case("default")

    def test_real_explicit_multiple_loading(self):
        self.run_case("multiple")

    def test_real_pickle_only_rejection(self):
        self.run_case("pickle")

    def test_real_generic_peft_configuration(self):
        self.run_case("generic")

    def test_real_subfolder_and_direct_adapter_loading(self):
        self.run_case("subfolder")


if __name__ == "__main__":
    unittest.main()