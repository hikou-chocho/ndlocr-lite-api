using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NdlocrLiteApi.Models;
using OpenCvSharp;
using OrtSessionOptions = Microsoft.ML.OnnxRuntime.SessionOptions;
using YamlDotNet.Serialization;

namespace NdlocrLiteApi.Services;

/// <summary>
/// DEIM + PARSEQ x3 を使用した OCR エンジン。
/// 起動時に Initialize() を一度呼び出してモデルをロードする。
/// </summary>
public class OcrEngine
{
    // --- パス定数 ---
    private static string SampleLibSrc =>
        Path.Combine(Directory.GetCurrentDirectory(), "sampleLib", "ndlocr-lite-1.1.0", "src");

    private static string ModelDir => Path.Combine(SampleLibSrc, "model");
    private static string ConfigDir => Path.Combine(SampleLibSrc, "config");

    private static string DeimModelPath  => Path.Combine(ModelDir, "deim-s-1024x1024.onnx");
    private static string Model30Path    => Path.Combine(ModelDir, "parseq-ndl-16x256-30-tiny-192epoch-tegaki3.onnx");
    private static string Model50Path    => Path.Combine(ModelDir, "parseq-ndl-16x384-50-tiny-146epoch-tegaki2.onnx");
    private static string Model100Path   => Path.Combine(ModelDir, "parseq-ndl-16x768-100-tiny-165epoch-tegaki2.onnx");
    private static string NdlMojiYaml   => Path.Combine(ConfigDir, "NDLmoji.yaml");

    // --- NDL クラス名（ndl.yaml より）---
    private static readonly string[] NdlClasses =
    {
        "text_block", "line_main", "line_caption", "line_ad", "line_note",
        "line_note_tochu", "block_fig", "block_ad", "block_pillar", "block_folio",
        "block_rubi", "block_chart", "block_eqn", "block_cfm", "block_eng",
        "block_table", "line_title"
    };

    // --- DEIM ---
    private InferenceSession? _deimSession;
    private int _deimInputH, _deimInputW;
    private string[] _deimInputNames = Array.Empty<string>();
    private bool _deimHasCharCounts;
    private const float ConfThreshold = 0.25f;

    // --- PARSEQ ---
    private ParseqModel? _rec30, _rec50, _rec100;

    // --- 文字セット ---
    private char[] _charList = Array.Empty<char>();

    /// <summary>
    /// 全モデルと文字セットを読み込む。起動時に一度だけ呼ぶ。
    /// </summary>
    public void Initialize()
    {
        Console.WriteLine("[INFO] Loading charset...");
        LoadCharset();

        Console.WriteLine("[INFO] Loading DEIM model...");
        LoadDeim();

        Console.WriteLine("[INFO] Loading PARSEQ models...");
        _rec30  = LoadParseq(Model30Path);
        _rec50  = LoadParseq(Model50Path);
        _rec100 = LoadParseq(Model100Path);

        Console.WriteLine("[INFO] All models loaded.");
    }

    // ==================== 公開メソッド ====================

    /// <summary>
    /// DEIM で画像内のテキスト行を検出する。
    /// line_* クラスのみ抽出し、高さが minHeight 以上のものを y 座標昇順で返す。
    /// </summary>
    public List<LineDetection> DetectLines(Mat matBgr, int minHeight)
    {
        var (imageTensor, sizeTensor, paddedSize) = PreprocessDeim(matBgr);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_deimInputNames[0], imageTensor),
        };
        if (_deimInputNames.Length >= 2)
            inputs.Add(NamedOnnxValue.CreateFromTensor(_deimInputNames[1], sizeTensor));

        using var outputs = _deimSession!.Run(inputs);

        return PostprocessDeim(outputs, paddedSize, minHeight);
    }

    /// <summary>
    /// PARSEQ カスケードで行画像を文字認識する。
    /// predCharCount をヒントに初期モデルを選択し、結果長でエスカレーション。
    /// </summary>
    public string Recognize(Mat cropBgr, float predCharCount)
    {
        string text;

        if (predCharCount <= 30)
        {
            text = RunParseq(_rec30!, cropBgr);
            if (text.Length >= 25)
                text = RunParseq(_rec50!, cropBgr);
            if (text.Length >= 45)
                text = RunParseq(_rec100!, cropBgr);
        }
        else if (predCharCount <= 50)
        {
            text = RunParseq(_rec50!, cropBgr);
            if (text.Length >= 45)
                text = RunParseq(_rec100!, cropBgr);
        }
        else
        {
            text = RunParseq(_rec100!, cropBgr);
        }

        return text;
    }

    // ==================== 内部ロード処理 ====================

    private void LoadCharset()
    {
        // YamlDotNet で NDLmoji.yaml をパース
        // charset_train を charList として使用（ocr.py:100 準拠）
        string yamlText = File.ReadAllText(NdlMojiYaml);
        var deserializer = new DeserializerBuilder().Build();
        var root = deserializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(yamlText);
        string charset = root["model"]["charset_train"];
        _charList = charset.ToCharArray();
    }

    private void LoadDeim()
    {
        var opts = new OrtSessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // シングルスレッド（仕様書 §8 準拠）
        opts.IntraOpNumThreads = 1;
        opts.InterOpNumThreads = 1;

        _deimSession = new InferenceSession(DeimModelPath, opts);

        _deimInputNames = _deimSession.InputNames.ToArray();

        // DEIM モデルの入力サイズを取得（動的次元の場合はデフォルト 1024 を使用）
        var firstInputMeta = _deimSession.InputMetadata[_deimInputNames[0]];
        var dims = firstInputMeta.Dimensions;
        _deimInputH = (dims.Length > 2 && dims[2] > 0) ? dims[2] : 1024;
        _deimInputW = (dims.Length > 3 && dims[3] > 0) ? dims[3] : 1024;

        _deimHasCharCounts = _deimSession.OutputNames.Count >= 4;

        // 診断: 入出力メタ情報をコンソールへ出力
        Console.WriteLine($"[DEIM] InputH={_deimInputH}, InputW={_deimInputW}");
        Console.WriteLine($"[DEIM] Inputs : {string.Join(", ", _deimSession.InputNames)}");
        Console.WriteLine($"[DEIM] Outputs: {string.Join(", ", _deimSession.OutputNames)}");
        foreach (var kv in _deimSession.OutputMetadata)
        {
            var d = string.Join("x", kv.Value.Dimensions);
            Console.WriteLine($"  [{kv.Key}] type={kv.Value.ElementType} dims=[{d}]");
        }
    }

    private ParseqModel LoadParseq(string modelPath)
    {
        var opts = new OrtSessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.IntraOpNumThreads = 1;
        opts.InterOpNumThreads = 1;

        var session = new InferenceSession(modelPath, opts);
        string inputName  = session.InputNames.First();
        string outputName = session.OutputNames.First();

        var dims = session.InputMetadata[inputName].Dimensions;
        // PARSEQ モデル名の命名: "16x{W}-{N}" → inputH=16, inputW=W
        int inputH = (dims.Length > 2 && dims[2] > 0) ? dims[2] : 16;
        int inputW = (dims.Length > 3 && dims[3] > 0) ? dims[3] : 384;

        return new ParseqModel(session, inputH, inputW, inputName, outputName);
    }

    // ==================== DEIM 前処理・後処理 ====================

    /// <summary>
    /// deim.py:preprocess に準拠。
    /// BGR Mat → パディング → リサイズ → ImageNet 正規化(RGB) → NCHW テンソル
    /// </summary>
    private (DenseTensor<float> imageTensor, DenseTensor<long> sizeTensor, int paddedSize)
        PreprocessDeim(Mat matBgr)
    {
        int paddedSize = Math.Max(matBgr.Rows, matBgr.Cols);

        // 正方形パディング（左上に貼り付け、残りは黒）
        using Mat padded = Mat.Zeros(paddedSize, paddedSize, MatType.CV_8UC3);
        matBgr.CopyTo(padded[new Rect(0, 0, matBgr.Cols, matBgr.Rows)]);

        // モデル入力サイズにリサイズ
        using Mat resizedBgr = new();
        Cv2.Resize(padded, resizedBgr, new Size(_deimInputW, _deimInputH));

        // BGR → RGB（deim.py は RGB入力を期待: 前処理でBGR反転なし、ImageNet RGB mean/std）
        using Mat resizedRgb = new();
        Cv2.CvtColor(resizedBgr, resizedRgb, ColorConversionCodes.BGR2RGB);

        // ImageNet 正規化 (RGB 順)
        float[] means = { 0.485f, 0.456f, 0.406f };
        float[] stds  = { 0.229f, 0.224f, 0.225f };

        float[] imageData = new float[3 * _deimInputH * _deimInputW];
        byte[] rawPixels = new byte[resizedRgb.Total() * resizedRgb.ElemSize()];
        Marshal.Copy(resizedRgb.Data, rawPixels, 0, rawPixels.Length);

        int stride = _deimInputW * 3;
        for (int c = 0; c < 3; c++)
        {
            int channelOffset = c * _deimInputH * _deimInputW;
            for (int y = 0; y < _deimInputH; y++)
            {
                for (int x = 0; x < _deimInputW; x++)
                {
                    // RGB Mat: rawPixels[y*stride + x*3 + 0]=R, [1]=G, [2]=B
                    float pixelVal = rawPixels[y * stride + x * 3 + c] / 255.0f;
                    imageData[channelOffset + y * _deimInputW + x] =
                        (pixelVal - means[c]) / stds[c];
                }
            }
        }

        var imageTensor = new DenseTensor<float>(imageData,
            new[] { 1, 3, _deimInputH, _deimInputW });

        // 第2入力: サイズテンソル [[inputH, inputW]] shape [1, 2]（deim.py:126）
        var sizeTensor = new DenseTensor<long>(
            new long[] { _deimInputH, _deimInputW }, new[] { 1, 2 });

        return (imageTensor, sizeTensor, paddedSize);
    }

    /// <summary>
    /// deim.py:postprocess に準拠。
    /// 出力テンソルを解析して LineDetection リストを返す。
    /// </summary>
    private List<LineDetection> PostprocessDeim(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int paddedSize, int minHeight)
    {
        // 出力順: class_ids, bboxes, scores, [char_counts]
        // 出力名と型を診断ログへ出力（デバッグ用）
        var outputList = outputs.ToList();
        for (int oi = 0; oi < outputList.Count; oi++)
        {
            var ov = outputList[oi];
            Console.WriteLine($"[DEIM out#{oi}] name={ov.Name} type={ov.ElementType}");
        }

        float[] scoresFlat = outputList[2].AsEnumerable<float>().ToArray();
        float[] bboxesFlat = outputList[1].AsEnumerable<float>().ToArray();

        // class_ids: モデルによって int64 または float32 で返ることがある
        // int(label) で使うので float として取得してから int に変換
        long[] classIdsFlat = GetClassIds(outputList[0]);

        float[] charCountsFlat;
        if (_deimHasCharCounts)
            charCountsFlat = GetCharCounts(outputList[3]);
        else
            charCountsFlat = Enumerable.Repeat(100.0f, scoresFlat.Length).ToArray();

        // n = スコアの件数（squeezeで1次元になったと仮定）
        int n = scoresFlat.Length;

        // スケール: deim.py:99-104 (縦横とも同じ paddedSize/inputWidth を使用)
        float scale = (float)paddedSize / _deimInputW;

        var detections = new List<LineDetection>();
        for (int i = 0; i < n && i < classIdsFlat.Length; i++)
        {
            float score = scoresFlat[i];
            if (score <= ConfThreshold) continue;

            int classIndex = (int)classIdsFlat[i] - 1; // deim.py:108 (-1 for 0-based)
            if (classIndex < 0 || classIndex >= NdlClasses.Length) continue;

            string className = NdlClasses[classIndex];

            // bboxes: [xmin, ymin, xmax, ymax] scaled to original image coords
            int x1 = (int)(bboxesFlat[i * 4 + 0] * scale);
            int y1 = (int)(bboxesFlat[i * 4 + 1] * scale);
            int x2 = (int)(bboxesFlat[i * 4 + 2] * scale);
            int y2 = (int)(bboxesFlat[i * 4 + 3] * scale);

            detections.Add(new LineDetection
            {
                X1 = x1, Y1 = y1, X2 = x2, Y2 = y2,
                ClassName = className,
                PredCharCount = charCountsFlat[i],
                Confidence = score
            });
        }

        return detections
            .Where(d => d.ClassName.StartsWith("line_") && d.H >= minHeight)
            .OrderBy(d => d.Y1)
            .ToList();
    }

    // ==================== PARSEQ 推論 ====================

    /// <summary>
    /// parseq.py:read に準拠。
    /// BGR Mat → 前処理 → ONNX 推論 → Greedy Decode → 文字列
    /// </summary>
    private string RunParseq(ParseqModel rec, Mat cropBgr)
    {
        var inputTensor = PreprocessParseq(rec, cropBgr);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(rec.InputName, inputTensor)
        };

        using var outputs = rec.Session.Run(inputs);

        // outputs shape: [1, seqLen, numClasses]
        var outputTensor = outputs.First().AsTensor<float>();
        int seqLen    = outputTensor.Dimensions[1];
        int numClasses = outputTensor.Dimensions[2];

        return GreedyDecode(outputTensor, seqLen, numClasses);
    }

    /// <summary>
    /// parseq.py:preprocess に準拠。
    /// BGR Mat → 回転（縦長の場合）→ リサイズ → 正規化 → NCHW テンソル
    ///
    /// parseq.py は RGB 入力を受け取り [:,:,::-1] で BGR に反転してから正規化する。
    /// C# では OpenCV の BGR Mat をそのまま使用すると同じ結果になる。
    /// </summary>
    private DenseTensor<float> PreprocessParseq(ParseqModel rec, Mat cropBgr)
    {
        Mat workMat = cropBgr;
        bool needDispose = false;

        // 縦長なら 90° 回転（parseq.py:51）
        if (workMat.Rows > workMat.Cols)
        {
            Mat rotated = new();
            Cv2.Rotate(workMat, rotated, RotateFlags.Rotate90Counterclockwise);
            workMat = rotated;
            needDispose = true;
        }

        // リサイズ
        using Mat resized = new();
        Cv2.Resize(workMat, resized, new Size(rec.InputW, rec.InputH));
        if (needDispose) workMat.Dispose();

        // BGR → 正規化: 2*(x/255 - 0.5) を全チャンネルに適用
        float[] imageData = new float[3 * rec.InputH * rec.InputW];
        byte[] rawPixels = new byte[resized.Total() * resized.ElemSize()];
        Marshal.Copy(resized.Data, rawPixels, 0, rawPixels.Length);

        int stride = rec.InputW * 3;
        for (int c = 0; c < 3; c++)
        {
            int channelOffset = c * rec.InputH * rec.InputW;
            for (int y = 0; y < rec.InputH; y++)
            {
                for (int x = 0; x < rec.InputW; x++)
                {
                    // BGR Mat: c=0→B, c=1→G, c=2→R
                    float pixelVal = rawPixels[y * stride + x * 3 + c] / 255.0f;
                    imageData[channelOffset + y * rec.InputW + x] =
                        2.0f * (pixelVal - 0.5f);
                }
            }
        }

        return new DenseTensor<float>(imageData, new[] { 1, 3, rec.InputH, rec.InputW });
    }

    /// <summary>
    /// parseq.py:68-72 に準拠の Greedy Decode。
    /// 各タイムステップで確率最大クラスを選択し、インデックス 0（stop/blank）で停止。
    /// </summary>
    private string GreedyDecode(Tensor<float> output, int seqLen, int numClasses)
    {
        var sb = new System.Text.StringBuilder(seqLen);

        for (int t = 0; t < seqLen; t++)
        {
            // argmax across class dimension
            int bestClass = 0;
            float bestVal = float.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                float val = output[0, t, c];
                if (val > bestVal)
                {
                    bestVal = val;
                    bestClass = c;
                }
            }

            // stop token は index 0（parseq.py:69）
            if (bestClass == 0) break;

            // charList は 1-indexed（parseq.py:72: charlist[i - 1]）
            int charIdx = bestClass - 1;
            if (charIdx >= 0 && charIdx < _charList.Length)
                sb.Append(_charList[charIdx]);
        }

        return sb.ToString();
    }

    // ==================== ユーティリティ ====================

    /// <summary>
    /// DEIM の class_ids テンソルを long[] に変換する。
    /// モデルによって int64 / int32 / float32 で返ることがあるため型を判定して対応する。
    /// </summary>
    private static long[] GetClassIds(DisposableNamedOnnxValue ov)
    {
        return ov.ElementType switch
        {
            TensorElementType.Int64 => ov.AsEnumerable<long>().ToArray(),
            TensorElementType.Int32 => ov.AsEnumerable<int>().Select(v => (long)v).ToArray(),
            TensorElementType.Float => ov.AsEnumerable<float>().Select(v => (long)v).ToArray(),
            _ => throw new InvalidOperationException(
                     $"Unsupported class_ids element type: {ov.ElementType}")
        };
    }

    private static float[] GetCharCounts(DisposableNamedOnnxValue ov)
    {
        return ov.ElementType switch
        {
            TensorElementType.Float => ov.AsEnumerable<float>().ToArray(),
            TensorElementType.Int64 => ov.AsEnumerable<long>().Select(v => (float)v).ToArray(),
            TensorElementType.Int32 => ov.AsEnumerable<int>().Select(v => (float)v).ToArray(),
            _ => throw new InvalidOperationException(
                     $"Unsupported char_count element type: {ov.ElementType}")
        };
    }

    // ==================== 内部型 ====================

    private record ParseqModel(
        InferenceSession Session,
        int InputH,
        int InputW,
        string InputName,
        string OutputName);
}
