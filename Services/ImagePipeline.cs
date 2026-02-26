using OpenCvSharp;

namespace NdlocrLiteApi.Services;

/// <summary>
/// 画像の検証・前処理を担当するユーティリティクラス。
/// </summary>
public static class ImagePipeline
{
    private const long MaxFileSizeBytes = 10L * 1024 * 1024; // 10 MB
    private const int MaxDimensionPx = 3000;

    private static readonly HashSet<string> AllowedExtensions =
        new(StringComparer.OrdinalIgnoreCase) { ".png", ".jpg", ".jpeg" };

    /// <summary>
    /// バイト列から画像を読み込み、サイズ・形式・解像度を検証する。
    /// 検証 OK なら BGR Mat を返す。失敗なら null を返す。
    /// </summary>
    public static Mat? ValidateImage(byte[] bytes, string fileName)
    {
        // ファイルサイズチェック
        if (bytes.Length > MaxFileSizeBytes)
            return null;

        // 拡張子チェック
        var ext = Path.GetExtension(fileName);
        if (!AllowedExtensions.Contains(ext))
            return null;

        // 画像デコード（OpenCV: BGR 形式）
        Mat mat = Cv2.ImDecode(bytes, ImreadModes.Color);
        if (mat.Empty())
            return null;

        // 解像度チェック
        if (mat.Cols > MaxDimensionPx || mat.Rows > MaxDimensionPx)
        {
            mat.Dispose();
            return null;
        }

        return mat;
    }

    /// <summary>
    /// 画像が白紙かどうかを判定する。
    /// Otsu の二値化で黒画素率が 0.1% 未満なら白紙とみなす。
    /// </summary>
    public static bool IsBlank(Mat mat)
    {
        using Mat gray = new();
        Cv2.CvtColor(mat, gray, ColorConversionCodes.BGR2GRAY);

        using Mat binary = new();
        // THRESH_BINARY: テキスト=黒(0)、背景=白(255)
        Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.Otsu | ThresholdTypes.Binary);

        // 黒画素（テキスト画素）数
        int totalPixels = binary.Rows * binary.Cols;
        // CountNonZero は 0以外（白）の数なので、黒は total - nonzero
        int blackPixels = totalPixels - Cv2.CountNonZero(binary);

        double blackRate = (double)blackPixels / totalPixels;
        return blackRate < 0.001;
    }
}
