using Microsoft.AspNetCore.Http.Features;
using NdlocrLiteApi.Services;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddSingleton<OcrEngine>();

// multipart/form-data のサイズ上限を 11MB に設定（10MB 制限チェックは アプリ内で行う）
builder.WebHost.ConfigureKestrel(o => o.Limits.MaxRequestBodySize = 11 * 1024 * 1024);
builder.Services.Configure<FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 11 * 1024 * 1024;
});

var app = builder.Build();

// 起動時にモデルをロード（シングルトン OcrEngine の初期化）
app.Services.GetRequiredService<OcrEngine>().Initialize();

// wwwroot 配下の静的ファイルを配信
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapControllers();
app.Run("http://localhost:8000");
