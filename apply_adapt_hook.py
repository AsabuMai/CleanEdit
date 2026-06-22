from pathlib import Path
def insert(path, pred, block):
    lines = Path(path).read_text().split("\n")
    if any("ADAPT_OVERRIDE_JSON" in l for l in lines):
        print("already hooked", path); return
    idx = next(i for i, l in enumerate(lines) if pred(l))
    lines[idx:idx] = block
    Path(path).write_text("\n".join(lines))
    print("inserted into", path, "before line", idx)
sd3_block = [
 "    _adapt = __import__(\"os\").environ.get(\"ADAPT_OVERRIDE_JSON\")",
 "    if _adapt:",
 "        for _fl, _vl in json.loads(_adapt).get(k, {}).items():",
 "            if _fl in argv: argv[argv.index(_fl) + 1] = str(_vl)",
 "            else: argv += [_fl, str(_vl)]",
]
flux_block = [
 "    _adapt = __import__(\"os\").environ.get(\"ADAPT_OVERRIDE_JSON\")",
 "    if _adapt:",
 "        for _fl, _vl in json.loads(_adapt).get(kind_of(e), {}).items():",
 "            if _fl in a: a[a.index(_fl) + 1] = str(_vl)",
 "            else: a += [_fl, str(_vl)]",
]
insert("scripts/pie_sd3_batch_kindaware.py", lambda l: l.lstrip().startswith("sys.argv=[\"run_edit_sd3.py\"]"), sd3_block)
insert("scripts/pie_batch_flux_kindaware.py", lambda l: l.strip() == "return a", flux_block)
