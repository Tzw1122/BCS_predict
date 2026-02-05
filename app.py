"""
基于SMILES预测药物性质的Flask应用 - 简化版
"""
from flask import Flask, render_template, request
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Draw import MolDraw2DCairo as MolDrawer
import sys
import os
import io
import base64
from datetime import datetime

def generate_molecule_image(smiles):
    """使用RDKit生成分子结构图像，返回base64编码"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 设置图像大小和样式
        drawer = MolDrawer(400, 300)
        drawer.SetFontSize(0.8)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        # 将图像转换为base64
        img_data = drawer.GetDrawingText()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        return img_base64
    except Exception as e:
            print(f"生成分子图像失败: {e}")
            return None

# 初始化Flask应用
app = Flask(__name__)

# 定义模型路径
LOG_S_PATH = "./LogS prediction/LightGBM Model（自动提取特征）"
LOG_PAPP_PATH = "./LogPapp prediction/XGBoost Model"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取输入参数
        smiles = request.form.get('smiles', '').strip()
        dose_str = request.form.get('dose', '50').strip()

        # 验证SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return render_template('result.html',
                                 error="无效的SMILES字符串",
                                 smiles=smiles,
                                 dose=dose_str)

        # 解析剂量
        try:
            dose = float(dose_str)
            if dose <= 0:
                dose = 50.0
        except:
            dose = 50.0

        # 生成分子图像
        mol_image = generate_molecule_image(smiles)


        # 计算分子量
        MW = float(Descriptors.MolWt(mol))

        """
        预测logS
        """
        sys.path.insert(0, LOG_S_PATH)
        try:
            from logS_feature_extractor import calculate_molecular_descriptors
            from logS_process_dataset import DataPreprocessor as LogS_Preprocessor
            from logS_model import LightGBM as LogS_Model

            # 提取特征
            features_df = calculate_molecular_descriptors([smiles])
            if features_df is None:
                return render_template('result.html',
                                     error="无法提取logS预测的特征",
                                     smiles=smiles,
                                     dose=dose)

            # 预处理特征
            preprocessor = LogS_Preprocessor()
            preprocessor.load_preprocessor('logS_preprocessor.pkl')
            features_processed = preprocessor.transform(features_df)

            # 定义LightGBM参数（必须提供所有参数）
            logS_params = {
                'n_estimators': 300,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': 12,
                'min_child_samples': 25,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'reg_alpha': 2.0,
                'reg_lambda': 0.5,
                'random_state': 42,
                'n_jobs': -1
            }

            # 加载模型并进行预测（必须传递参数！）
            model = LogS_Model(**logS_params)
            model.load_model('final_model.pkl')
            logS = float(model.predict(features_processed)[0])

        except Exception as e:
            sys.path.pop(0)
            return render_template('result.html',
                                 error=f"logS预测失败: {str(e)}",
                                 smiles=smiles,
                                 dose=dose)
        sys.path.pop(0)

        """
        预测logPapp
        """
        sys.path.insert(0, LOG_PAPP_PATH)
        try:
            from logPapp_feature_extractor import extract_descriptors
            from logPapp_process_dataset import DataPreprocessor as LogPapp_Preprocessor
            from logPapp_model import XGBoost as LogPapp_Model

            # 提取特征
            features_df = extract_descriptors([smiles])
            if features_df is None:
                return render_template('result.html',
                                     error="无法提取logPapp预测的特征",
                                     smiles=smiles,
                                     dose=dose)

            # 预处理特征
            preprocessor = LogPapp_Preprocessor()
            preprocessor.load_preprocessor('logPapp_preprocessor.pkl')
            features_processed = preprocessor.transform(features_df)

            # 定义XGBoost参数（必须提供所有参数）
            logPapp_params = {
                'n_estimators': 300,
                'learning_rate': 0.1,
                'max_depth': 12,
                'min_child_weight': 25,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'reg_alpha': 2.0,
                'reg_lambda': 0.5,
                'random_state': 21,
                'nthread': -1
            }

            # 加载模型并进行预测（必须传递参数！）
            model = LogPapp_Model(**logPapp_params)
            model.load_model('final_model.json')
            logPapp = float(model.predict(features_processed)[0])

        except Exception as e:
            sys.path.pop(0)
            return render_template('result.html',
                                 error=f"logPapp预测失败: {str(e)}",
                                 smiles=smiles,
                                 dose=dose)
        sys.path.pop(0)

        """
        计算D₀和BCS分类
        """
        try:
            # 确保所有变量都是数字
            logS = float(logS)
            MW = float(MW)
            dose = float(dose)
            logPapp = float(logPapp)

            # 计算D₀
            S = (10 ** logS) * MW  # 溶解度，单位: mg/mL
            D0 = dose / (250 * S)


            # 判断BCS类别
            if D0 <= 1:
                if logPapp >= -5.15:
                    BCS = "I类"
                else:
                    BCS = "III类"
            else:
                if logPapp >= -5.15:
                    BCS = "II类"
                else:
                    BCS = "IV类"

            # 获取当前时间
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 渲染结果页面
            return render_template('result.html',
                                           smiles=smiles,
                                           dose=f"{dose:.1f}",
                                           logS=f"{logS:.3f}",
                                           MW=f"{MW:.2f}",
                                           D0=f"{D0:.3f}",
                                           logPapp=f"{logPapp:.3f}",
                                           BCS=BCS,
                                           mol_image=mol_image,
                                           now=now)

        except Exception as e:
            return render_template('result.html',
                                 error=f"参数计算失败: {str(e)}",
                                 smiles=smiles,
                                 dose=dose)

    except Exception as e:
        return render_template('result.html',
                             error=f"预测过程中发生错误: {str(e)}",
                             smiles=smiles,
                             dose=dose_str if 'dose_str' in locals() else '50')
# 主函数
if __name__ == '__main__':
    # 本地开发时运行
    app.run(debug=True, host='0.0.0.0', port=5000)