import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import utils

import argparse

def load_args():
    parser = argparse.ArgumentParser(description = "a")
    parser.add_argument("img",help ="img name in picture folder")
    args = parser.parse_args()
    print(args.img)
    return args


def return_class():
    # cl = utils.return_class()
    cl = ['3Dランド', '75m', 'WiiFitスタジオ', 'いかだと滝', 'いにしえっぽい王国', 'いにしえの王国', 'いにしえの王国USA', 'すま村', 'すれちがい伝説', 'とある星', 'はじまりの塔', 'アンブラの時計塔', 'イッシュポケモンリーグ', 'ウィンディヒル', 'ウーフーアイランド', 'エイトクロスサーキット', 'エレクトロプランクトン', 'エンジェランド', 'オネット', 'オルディン大橋', 'カロスポケモンリーグ', 'ガウル平原', 'グリーングリーンズ', 'グリーンヒルゾーン', 'グレートベイ', 'ゲルドの谷', 'ゲーマー', 'コトブキランド', 'コンゴジャングル', 'シャドーモセス島', 'ジャングルガーデン', 'スカイロフト', 'スーパーしあわせのツリー', 'スーパーマリオメーカー', 'タチウオパーキング', 'ダックハント', 'テンガンざんやりのはしら', 'トモダチコレクション', 'トレーニング', 'ドラキュラ城', 'ドルピックタウン', 'ニュードンク市庁舎', 'ニューポークシティ', 'ノルフェア', 'ハイラル城', 'バルーンファイト', 'パイロットウィングス', 'パックランド', 'ビッグブルー', 'ピクトチャット2', 'ピーチ城', 'ピーチ城上空', 'フェリア闘技場', 'フォーサイド', 'フラットゾーンX', 'フリゲートオルフェオン', 'ブリンスタ', 'ブリンスタ深部', 'プププランド', 'プププランドGB', 'プリズムタワー', 'ペーパーマリオ', 'ポケモンスタジアム', 'ポケモンスタジアム２', 'ポートタウンエアロダイブ', 'マジカント', 'マリオUワールド', 'マリオギャラクシー', 'マリオサーキット', 'マリオブラザーズ', 'ミッドガル', 'ミュートシティSFC', 'メイドインワリオ', 'ヤマブキシティ', 'ヨッシーアイランド', 'ヨッシーストーリー', 'ヨースター島', 'ライラットクルーズ', 'ルイージマンション', 'レインボークルーズ', 'レッキングクルー', 'ワイリー基地', '再開の花園', '初期化爆弾の森', '夢の泉', '天空海', '子犬がいるリビング', '惑星コーネリア', '惑星ベノム', '戦場', '戦艦ハルバード', '攻城戦', '朱雀嬢', '村と街', '汽車', '洞窟大作戦', '海賊船', '特設リング', '神殿', '終点', '野原', '闘技場', '頂上']
    return cl

def load_weight():
    net = models.mobilenet_v2()
    num_features = 1280
    net.classifier[1] = nn.Linear(num_features, 103)

    net.load_state_dict(torch.load("weight/classificate.pth"))
    net = net.eval()
    return net


def loadpic(args):
    path = "picture/"+str(args.img)
    img = Image.open(path)
    w,h = img.size
    t = (0,0,w//2, h//2)
    # img = img.crop(t)
    img = img.resize((224,224)).convert('RGB') 
    return img 

def predict(args):
    net = load_weight()
    img = loadpic(args)
    classes = return_class()

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = test_transform(img)
    img = img.view(1,3,224,224)
    outputs = net(img)
    _, predicted = torch.max(outputs.data, 1)
    st = classes[predicted[0]]
    print(predicted)
    print(st)

if __name__=="__main__":
    # loadpic(load_args)
    predict(load_args())
