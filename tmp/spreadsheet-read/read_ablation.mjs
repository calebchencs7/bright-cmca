import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "/Users/haochen/Documents/Development/bright-cmca/Ablation experiment.xlsx";
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 12000,
});
console.log("===== SHEETS =====");
console.log(sheets.ndjson);

const matches = await workbook.inspect({
  kind: "match",
  searchTerm: "SiamAttnUNet",
  options: { useRegex: false, maxResults: 200 },
  maxChars: 30000,
});
console.log("===== SIAMATTNUNET MATCHES =====");
console.log(matches.ndjson);

for (const [sheetId, range] of [
  ["Overall", "A13:I19"],
  ["per-event mIoU", "M1:S16"],
  ["SiamAttnUnet Overall", "A1:I7"],
  ["SiamAttnUnet per-event", "A1:H14"],
]) {
  const table = await workbook.inspect({
    kind: "table",
    sheetId,
    range,
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 12,
    tableMaxCellChars: 120,
    maxChars: 30000,
  });
  console.log(`===== ${sheetId}!${range} =====`);
  console.log(table.ndjson);
}

await fs.mkdir("/Users/haochen/Documents/Development/bright-cmca/tmp/spreadsheet-read/renders", { recursive: true });
for (const sheet of workbook.worksheets.items) {
  const used = sheet.getUsedRange();
  if (!used) continue;
  const preview = await workbook.render({
    sheetName: sheet.name,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = sheet.name.replace(/[^A-Za-z0-9_-]+/g, "_");
  await fs.writeFile(
    `/Users/haochen/Documents/Development/bright-cmca/tmp/spreadsheet-read/renders/${safeName}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
