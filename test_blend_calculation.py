import numpy as np

def calculate_blend_with_background(A, B, sharpness=20):
    """
    A, B: 各画像の値 (0.0-1.0)
    returns: (background_weight, A_weight, B_weight)
    """
    max_val = max(A, B)
    
    # 1.0なら完全優先
    if max_val >= 1.0:
        if A >= 1.0 and B >= 1.0:
            return (0.0, 0.5, 0.5)
        elif A >= 1.0:
            return (0.0, 1.0, 0.0)
        else:  # B >= 1.0
            return (0.0, 0.0, 1.0)
    
    # 1.0未満：背景を指数関数的に減少
    background = (1 - max_val) ** sharpness
    
    # 残りをA:Bの比率で配分
    remaining = 1.0 - background
    total_ab = A + B
    
    if total_ab > 0:
        A_weight = remaining * (A / total_ab)
        B_weight = remaining * (B / total_ab)
    else:
        A_weight = 0.0
        B_weight = 0.0
    
    return (background, A_weight, B_weight)


def calculate_blend_custom(A, B):
    """
    カスタム計算式で要求仕様に近づける
    """
    max_val = max(A, B)
    
    # 1.0なら完全優先
    if max_val >= 1.0:
        if A >= 1.0 and B >= 1.0:
            return (0.0, 0.5, 0.5)
        elif A >= 1.0:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)
    
    # 背景の計算（カスタム式）
    if max_val <= 0.5:
        # 0.5以下：背景多め（約30%）
        background = 0.3 * (1 - max_val * 0.6)
    elif max_val <= 0.8:
        # 0.5-0.8：急激に減少
        background = 0.3 * ((0.8 - max_val) / 0.3) ** 2
    else:
        # 0.8-1.0：ほぼゼロに
        background = 0.01 * ((1.0 - max_val) / 0.2) ** 3
    
    # 残りを配分
    remaining = 1.0 - background
    total = A + B
    
    if total > 0:
        A_weight = remaining * (A / total)
        B_weight = remaining * (B / total)
    else:
        A_weight = B_weight = 0.0
    
    return (background, A_weight, B_weight)


def calculate_blend_advanced(A, B):
    """
    より精密な計算式
    """
    max_val = max(A, B)
    
    # 1.0なら完全優先
    if max_val >= 0.999:  # 浮動小数点誤差対策
        if A >= 0.999 and B >= 0.999:
            return (0.0, 0.5, 0.5)
        elif A >= 0.999:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)
    
    # シグモイド関数ベースの背景計算
    # max_valが高いほど急激に背景が減る
    x = max_val * 10 - 5  # 0.5を中心にスケーリング
    sigmoid = 1 / (1 + np.exp(-x))
    background = 0.35 * (1 - sigmoid)  # 最大35%の背景
    
    # 高い値での微調整
    if max_val > 0.85:
        # 0.85以上は背景をさらに減らす
        reduction = ((max_val - 0.85) / 0.15) ** 2
        background *= (1 - reduction * 0.95)
    
    # 残りを配分
    remaining = 1.0 - background
    total = A + B
    
    if total > 0:
        A_weight = remaining * (A / total)
        B_weight = remaining * (B / total)
    else:
        A_weight = B_weight = 0.0
    
    return (background, A_weight, B_weight)


def test_blend_functions():
    """
    各ブレンド関数をテスト
    """
    # テストケースと期待値
    test_cases = [
        # (A, B, 期待される背景, 期待されるA, 期待されるB)
        (1.0, 0.9, 0.0, 1.0, 0.0),
        (0.5, 0.5, 0.3, 0.35, 0.35),
        (0.9, 0.9, 0.001, 0.5, 0.5),
        (0.8, 0.9, 0.005, 0.45, 0.55),
    ]
    
    print("=" * 80)
    print("要求仕様:")
    print("-" * 80)
    for A, B, exp_bg, exp_A, exp_B in test_cases:
        print(f"A={A:.1f}, B={B:.1f} → 背景≈{exp_bg:.3f}, A≈{exp_A:.3f}, B≈{exp_B:.3f}")
    
    print("\n" + "=" * 80)
    print("テスト結果:")
    print("=" * 80)
    
    # 各関数でテスト
    functions = [
        ("指数関数版 (sharpness=20)", lambda a, b: calculate_blend_with_background(a, b, 20)),
        ("指数関数版 (sharpness=10)", lambda a, b: calculate_blend_with_background(a, b, 10)),
        ("カスタム版", calculate_blend_custom),
        ("シグモイド版", calculate_blend_advanced),
    ]
    
    for func_name, func in functions:
        print(f"\n{func_name}:")
        print("-" * 50)
        
        total_error = 0
        for A, B, exp_bg, exp_A, exp_B in test_cases:
            bg, a_w, b_w = func(A, B)
            
            # 誤差計算
            error_bg = abs(bg - exp_bg)
            error_A = abs(a_w - exp_A)
            error_B = abs(b_w - exp_B)
            total_error += error_bg + error_A + error_B
            
            print(f"A={A:.1f}, B={B:.1f} → 背景={bg:.4f}, A={a_w:.4f}, B={b_w:.4f}")
            print(f"  誤差: 背景={error_bg:.4f}, A={error_A:.4f}, B={error_B:.4f}")
        
        print(f"\n合計誤差: {total_error:.4f}")
    
    # 追加テスト：様々な値での挙動確認
    print("\n" + "=" * 80)
    print("追加テスト（様々な値）:")
    print("=" * 80)
    
    additional_tests = [
        (0.0, 0.0),   # 両方ゼロ
        (0.3, 0.7),   # 異なる低い値
        (0.6, 0.6),   # 中間値
        (0.95, 0.95), # 高い値
        (1.0, 1.0),   # 両方1.0
        (0.1, 0.9),   # 大きな差
    ]
    
    print("\nシグモイド版での結果:")
    print("-" * 50)
    for A, B in additional_tests:
        bg, a_w, b_w = calculate_blend_advanced(A, B)
        print(f"A={A:.2f}, B={B:.2f} → 背景={bg:.4f}, A={a_w:.4f}, B={b_w:.4f} (合計={bg+a_w+b_w:.4f})")
    
    # グラフ的な視覚化（テキストベース）
    print("\n" + "=" * 80)
    print("背景の減衰曲線（シグモイド版）:")
    print("=" * 80)
    
    for val in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        bg, _, _ = calculate_blend_advanced(val, val)
        bar_length = int(bg * 100)
        bar = "█" * bar_length
        print(f"max={val:.2f}: {bar:<35} {bg:.4f}")


if __name__ == "__main__":
    test_blend_functions()