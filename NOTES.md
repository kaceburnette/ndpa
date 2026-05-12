# NDPA Notes

## 2026-05-12 18:47:46 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Python package `datasets` is not installed. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.

## 2026-05-12 18:48:12 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:55:00 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:55:36 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:55:53 EDT - LMSYS initial run interpreter error

Initial LMSYS background run PID 24325 used Apple Python 3.9 and failed before benchmark execution:
```
Traceback (most recent call last):
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/runpy.py", line 197, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/Users/kaceburnette/Desktop/ndp/eval/lmsys_validation.py", line 22, in <module>
    from eval.conversation_eval import Conversation, cosine, tokenize
  File "/Users/kaceburnette/Desktop/ndp/eval/conversation_eval.py", line 28, in <module>
    from supabase import create_client
ModuleNotFoundError: No module named 'supabase'
```
Restarting with /opt/homebrew/bin/python3, where datasets was installed.

## 2026-05-12 18:56:08 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 18:56:23 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Dataset 'lmsys/lmsys-chat-1m' is a gated dataset on the Hub. You must be authenticated to access it.. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.

## 2026-05-12 19:06:39 EDT - LMSYS gated dataset blocked

Full LMSYS validation run PID 24743 installed datasets successfully but HuggingFace denied unauthenticated access to lmsys/lmsys-chat-1m:
```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Wrote /Users/kaceburnette/Desktop/ndp/eval/lmsys_results.json
blocked Dataset 'lmsys/lmsys-chat-1m' is a gated dataset on the Hub. You must be authenticated to access it.
```
Continuing with LongMemEval and LoCoMo.

## 2026-05-12 19:28:43 EDT - LoCoMo public dataset size

Requested LoCoMo runner expected 50 conversations, but the maintained Snap/USC public file at https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json contains 10 conversations. The runner scores the official public locomo10.json file and records the actual dataset size in eval/locomo_results.json.

## 2026-05-12 19:28:45 EDT - LMSYS validation blocked

LMSYS-Chat-1M validation could not run automatically. Reason: Dataset 'lmsys/lmsys-chat-1m' is a gated dataset on the Hub. You must be authenticated to access it.. The dataset is gated on HuggingFace and distributed as parquet; this repo intentionally does not add datasets/pyarrow as required dependencies. Provide --input with an accepted/exported JSON or JSONL sample, or install datasets after accepting the license.

## 2026-05-12 19:49:37 EDT - LongMemEval timeout

LongMemEval run PID 51970 failed during concurrent ingestion:
```
           ~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/ssl.py", line 1138, in read
    return self._sslobj.read(len, buffer)
           ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
TimeoutError: The read operation timed out

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 157, in <module>
    main()
    ~~~~^^
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 151, in main
    results = run(args)
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 100, in run
    future.result()
    ~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/_base.py", line 443, in result
    return self.__get_result()
           ~~~~~~~~~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/_base.py", line 395, in __get_result
    raise self._exception
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/thread.py", line 86, in run
    result = ctx.run(self.task)
  File "/opt/homebrew/Cellar/python@3.14/3.14.4/Frameworks/Python.framework/Versions/3.14/lib/python3.14/concurrent/futures/thread.py", line 73, in run
    return fn(*args, **kwargs)
  File "/Users/kaceburnette/Desktop/ndp/eval/longmemeval_runner.py", line 92, in ingest_session
    client.log_events(session_id, events, end_user_id=end_user_id)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/kaceburnette/Desktop/ndp/sdk/python/ndpa/client.py", line 99, in log_events
    self._send(payload)
    ~~~~~~~~~~^^^^^^^^^
  File "/Users/kaceburnette/Desktop/ndp/sdk/python/ndpa/client.py", line 148, in _send
    self._post_path("/events", payload)
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/Users/kaceburnette/Desktop/ndp/sdk/python/ndpa/client.py", line 171, in _post_path
    raise NDPAError(f"NDPA request failed: {e}") from e
ndpa.client.NDPAError: NDPA request failed: The read operation timed out
```
Adding retry handling and restarting LongMemEval with a longer timeout.
