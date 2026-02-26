using Microsoft.AspNetCore.Mvc;
using NdlocrLiteApi.Models;
using NdlocrLiteApi.Services;
using OpenCvSharp;

namespace NdlocrLiteApi.Controllers;

[ApiController]
[Route("v1")]
public class OcrController(OcrEngine engine, ILogger<OcrController> logger) : ControllerBase
{
    [HttpPost("ocr")]
    [Consumes("multipart/form-data")]
    public async Task<IActionResult> Ocr(
        IFormFile file,
        [FromForm] int minLineHeightPx = 8)
    {
        // バイト列として読み込み
        await using var ms = new MemoryStream();
        await file.CopyToAsync(ms);
        byte[] bytes = ms.ToArray();

        // 画像検証（サイズ・形式・解像度）
        using Mat? mat = ImagePipeline.ValidateImage(bytes, file.FileName);
        if (mat is null)
            return BadRequest(new { error = "image_too_large" });

        // 白紙チェック（黒画素率 < 0.1%）
        if (ImagePipeline.IsBlank(mat))
            return BadRequest(new { error = "blank_image" });

        // DEIM による行検出
        List<LineDetection> lineDets;
        try
        {
            lineDets = engine.DetectLines(mat, minLineHeightPx);
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "[DetectLines] 推論エラー");
            return StatusCode(500, new { error = "inference_error" });
        }

        if (lineDets.Count == 0)
            return BadRequest(new { error = "no_lines_detected" });

        // 各行を OCR
        var lineResults = new List<LineResult>();
        int imgH = mat.Rows;
        int imgW = mat.Cols;

        try
        {
            foreach (var det in lineDets)
            {
                // 境界クランプ
                int x1 = Math.Max(0, det.X1);
                int y1 = Math.Max(0, det.Y1);
                int x2 = Math.Min(imgW, det.X2);
                int y2 = Math.Min(imgH, det.Y2);
                int w  = x2 - x1;
                int h  = y2 - y1;

                if (w <= 0 || h <= 0) continue;

                using var crop = new Mat(mat, new Rect(x1, y1, w, h));
                string text = engine.Recognize(crop, det.PredCharCount);

                lineResults.Add(new LineResult(text, new BBox(x1, y1, w, h)));
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "[Recognize] 推論エラー");
            return StatusCode(500, new { error = "inference_error" });
        }

        if (lineResults.Count == 0)
            return BadRequest(new { error = "no_lines_detected" });

        string fullText = string.Join("\n", lineResults.Select(l => l.Text));
        return Ok(new OcrResponse(fullText, lineResults));
    }
}
