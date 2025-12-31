#!/usr/bin/env node
const { parse } = require("@babel/parser");

let code = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (c) => (code += c));
process.stdin.on("end", () => {
  if (!code.trim()) process.exit(1);
  try {
    parse(code, {
      sourceType: "unambiguous",
      allowReturnOutsideFunction: true,
      allowAwaitOutsideFunction: true,
      plugins: [
        "jsx", "typescript",
        "classProperties", "classPrivateProperties", "classPrivateMethods",
        ["decorators", { decoratorsBeforeExport: true }],
        "dynamicImport", "objectRestSpread",
        "optionalCatchBinding", "optionalChaining",
        "nullishCoalescingOperator", "topLevelAwait",
        "exportDefaultFrom", "exportNamespaceFrom",
        "logicalAssignment",
      ],
    });
    process.exit(0);
  } catch {
    process.exit(1);
  }
});
