using System.Text.Json.Serialization;

namespace NdlocrLiteApi.Models;

public record OcrResponse(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("lines")] List<LineResult> Lines
);

public record LineResult(
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("bbox")] BBox Bbox
);

public record BBox(
    [property: JsonPropertyName("x")] int X,
    [property: JsonPropertyName("y")] int Y,
    [property: JsonPropertyName("w")] int W,
    [property: JsonPropertyName("h")] int H
);
