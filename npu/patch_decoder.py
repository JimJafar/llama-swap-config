import sys, openvino as ov
xml, outxml = sys.argv[1], sys.argv[2]
core = ov.Core()
m = core.read_model(xml)
done = 0
for g in m.get_ordered_ops():
    if g.get_type_name() == "Greater":
        src = g.input(0).get_source_output().get_node()
        if src.get_type_name() == "Range":
            conv = ov.opset1.convert(src.output(0), ov.Type.i64)
            conv.set_friendly_name(src.get_friendly_name() + "/mask_conv")
            g.input(0).replace_source_output(conv.output(0))
            done += 1
print("patched Greater nodes:", done)
ov.save_model(m, outxml, compress_to_fp16=False)
print("saved to", outxml)
