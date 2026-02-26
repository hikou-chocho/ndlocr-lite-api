namespace NdlocrLiteApi.Models;

/// <summary>DEIM による行検出結果（内部用）</summary>
public class LineDetection
{
    public int X1 { get; init; }
    public int Y1 { get; init; }
    public int X2 { get; init; }
    public int Y2 { get; init; }
    public int W => X2 - X1;
    public int H => Y2 - Y1;
    public string ClassName { get; init; } = string.Empty;
    public float PredCharCount { get; init; }
    public float Confidence { get; init; }
}
