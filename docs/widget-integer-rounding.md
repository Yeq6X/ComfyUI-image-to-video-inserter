# ComfyUI Widget Integer Rounding Solution

## 問題

ComfyUIの`addWidget`で`precision: 0`を設定して整数表示にしても、UIの矢印ボタンを使用すると内部的には0.1ずつ値が変更され、期待通りに1ずつ増減しない問題があります。

例：
```javascript
// これだけでは不十分
this.addWidget("number", "frame_1", 10, null, {"min": 0, "max": 10000, "step": 1, "precision": 0});
```

## 解決方法

ウィジェット作成後にカスタムコールバック関数を設定し、値変更時に適切な丸め処理を行います。

### 実装例

```javascript
// ウィジェットを作成
const widget = this.addWidget("number", "frame_1", 10, null, {
    "min": 0, 
    "max": 10000, 
    "step": 1, 
    "precision": 0
});

// 既存のコールバックを保存
const originalCallback = widget.callback;

// カスタム丸め処理を追加
widget.callback = function() {
    // 元のコールバックがある場合は実行
    if (originalCallback) originalCallback.apply(this, arguments);
    
    // 小数点以下を取得
    const decimal = this.value - Math.floor(this.value);
    
    // カスタム丸めロジック:
    // 0.5以下なら切り上げ、0.5より大きいなら切り下げ
    const result = decimal <= 0.5 ? Math.ceil(this.value) : Math.floor(this.value);
    
    // 値が変わった場合のみ更新
    if (result !== this.value) {
        this.value = result;
    }
};
```

### 丸めロジックの説明

この実装では、以下のロジックで整数値に丸めます：

- **小数点以下 ≤ 0.5**: `Math.ceil()`で切り上げ
- **小数点以下 > 0.5**: `Math.floor()`で切り下げ

これにより、UIの矢印ボタンを押すたびに値が1ずつ増減するようになります。

### 動作例

```
5.0 → 5.1 (小数点以下 0.1 ≤ 0.5) → Math.ceil(5.1) = 6
6.0 → 5.9 (小数点以下 0.9 > 0.5) → Math.floor(5.9) = 5
```

### 動的ウィジェット生成時の適用

ループで複数のウィジェットを生成する場合も同様に適用できます：

```javascript
for(let i = 1; i <= count; i++) {
    const frameWidget = this.addWidget("number", `frame_${i}`, i * 10, null, {
        "min": 0, "max": 10000, "step": 1, "precision": 0
    });
    
    const originalCallback = frameWidget.callback;
    frameWidget.callback = function() {
        if (originalCallback) originalCallback.apply(this, arguments);
        const decimal = this.value - Math.floor(this.value);
        const result = decimal <= 0.5 ? Math.ceil(this.value) : Math.floor(this.value);
        if (result !== this.value) {
            this.value = result;
        }
    };
}
```

## 注意事項

- この方法はComfyUIのLiteGraphウィジェットシステムに依存しています
- 元のコールバック関数がある場合は必ず保存・実行してください
- 値の更新は無限ループを避けるため、値が実際に変わった場合のみ行ってください

## 関連技術

- **LiteGraph**: ComfyUIで使用されているグラフライブラリ
- **Widget Callback**: ウィジェットの値変更時に呼び出される関数
- **ComfyUI Extension**: JavaScriptでのカスタムノード開発