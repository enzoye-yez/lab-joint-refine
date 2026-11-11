"""Run developer tests with external socket connections forbidden; no provider credentials."""
import argparse
import ipaddress
import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/"src"),str(PROJECT/"tests")]

def deny_external(event,args):
    if event=="socket.connect":
        address=args[1]
        try: allowed=ipaddress.ip_address(address[0]).is_loopback
        except (ValueError,TypeError,IndexError): allowed=False
        if not allowed: raise RuntimeError("Offline verification forbids external network connections")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules",nargs="*")
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    for key in list(os.environ):
        if "API_KEY" in key: os.environ.pop(key)
    tmp=PROJECT/".cache/offline-test-temp";tmp.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(tmp)
    sys.addaudithook(deny_external)
    from asset_prep.storage import file_hash,atomic_json
    from asset_prep.implementation import source_manifest,implementation_digest,dependency_versions
    from datetime import datetime,timezone
    def sources():
        paths=[*PROJECT.joinpath("src").rglob("*.py"),*PROJECT.joinpath("tests").rglob("*.py"),*PROJECT.joinpath("scripts").glob("*.py"),PROJECT/"run.ps1"]
        return {p.relative_to(PROJECT).as_posix():file_hash(p) for p in paths if p.is_file()}
    before=sources();started=datetime.now(timezone.utc).isoformat()
    if args.receipt and args.receipt.exists():raise ValueError("Verification receipts are immutable; choose a new path")
    suite=(unittest.defaultTestLoader.loadTestsFromNames(args.modules) if args.modules else
           unittest.defaultTestLoader.discover(str(PROJECT/"tests")))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    after=sources();unchanged=before==after
    receipt={"status":"pass" if result.wasSuccessful() and unchanged else "failed",
             "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
             "skipped":[{"test":str(t),"reason":r} for t,r in result.skipped],
             "unexpected_successes":[str(t) for t in result.unexpectedSuccesses],
             "modules":args.modules or ["all offline tests"],"network_policy":"external socket connections denied",
             "model_api_calls":0,"started_utc":started,"finished_utc":datetime.now(timezone.utc).isoformat(),
             "source_unchanged":unchanged,"implementation":implementation_digest(source_manifest(PROJECT)),
             "dependencies":dependency_versions(),"sources_before":before,"sources_after":after}
    if args.receipt:atomic_json(args.receipt,receipt)
    if not unchanged:print("Verification source changed during tests; receipt cannot certify this run",file=sys.stderr)
    raise SystemExit(0 if receipt["status"]=="pass" else 1)
