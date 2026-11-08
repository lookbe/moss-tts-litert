"""Operator counts per signature of a .tflite (total and by op type).

  .venv\\Scripts\\python tests\\opcount.py models\\lm_gpu_i8.tflite [signature]
"""
import collections
import sys

from ai_edge_litert import schema_py_generated as S

NAMES = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}


def main():
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else None
    with open(path, "rb") as f:
        head = f.read(64 << 20)          # the flatbuffer is small; weights live after it
    m = S.Model.GetRootAsModel(head, 0)
    codes = [m.OperatorCodes(i).BuiltinCode() for i in range(m.OperatorCodesLength())]
    for si in range(m.SignatureDefsLength()):
        sd = m.SignatureDefs(si)
        key = sd.SignatureKey().decode()
        if want and key != want:
            continue
        sg = m.Subgraphs(sd.SubgraphIndex())
        c = collections.Counter(NAMES.get(codes[sg.Operators(i).OpcodeIndex()], "?") for i in range(sg.OperatorsLength()))
        top = ", ".join(f"{k} {v}" for k, v in c.most_common(12))
        print(f"{key}: {sg.OperatorsLength()} ops  ({top})")


if __name__ == "__main__":
    main()
