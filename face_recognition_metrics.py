import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
#from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from backbones import get_model
from argparse import ArgumentParser
from pathlib import Path

from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report

@torch.no_grad()
def inference(net, img):
    feat = net(img).numpy()
    return feat

def detect_face(img):
    pass

@torch.no_grad()
def get_similarity_scores(model, genuine_pairs, imposter_pairs):
    model.eval()
    genuine_scores = []
    imposter_scores = []

    for img1, img2, is_genuine in tqdm((genuine_pairs + imposter_pairs)):
        try:
            score = compute_similarity(model, img1, img2)
            if is_genuine:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)
        except Exception as e:
            print(e)
            pass

    return genuine_scores, imposter_scores


def preprocess_image(img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    return img


def compute_similarity(model, img1, img2):
    # Process the images through your model to get features
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)
    features1 = model(img1)
    features2 = model(img2)

    # Compute the cosine similarity
    similarity = cosine_similarity(features1.cpu().numpy().reshape(1, -1),
                                   features2.cpu().numpy().reshape(1, -1))[0][0]
    return similarity

def do_it(model_path, genuine_pairs_path, imposter_pairs_path, output, weights="", label=""):
    name = "mbf"
    net = get_model(name, fp16=False)
    threshold = 0.363
    net.load_state_dict(torch.load(str(model_path)))
    if weights.name.endswith(".npz"):
        data = np.load(Path(weights))
        genuine_scores, imposter_scores = data["genuine_scores"], data["imposter_scores"]
    else:
        genuine_scores, imposter_scores = get_similarity_scores(net, parse_list(genuine_pairs_path),
                                                            parse_list(imposter_pairs_path))
        with open(Path(output) / f"{label}_genuine_scores.txt", "w") as f:
            scores = '\n'.join([str(x) for x in genuine_scores])
            f.write(scores)
        with open(Path(output) / f"{label}_imposter_scores.txt", "w") as f:
            scores = '\n'.join([str(x) for x in imposter_scores])
            f.write(scores)
        #genuine_scores = [1 if x >= threshold else 0 for x in genuine_scores]
        #imposter_scores = [1 if x >= threshold else 0 for x in imposter_scores]
        np.savez(Path(output)/ f"{label}_scores.npz",
             genuine_scores=genuine_scores,
             imposter_scores=imposter_scores)
    scores = np.concatenate([genuine_scores, imposter_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    ## Save FPR, TPR, and thresholds for later use
    #np.savez(Path(genuine_pairs_path).with_name(f"{label}_roc.npz"),
    #         fpr=fpr,
    #         tpr=tpr,
    #         thresholds=thresholds)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(Path(output)/ f"{label}_roc_curve.png")
    plt.close()



    frr = 1 - tpr
    far= fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((frr - far)))]
    eer_index = np.nanargmin(np.absolute(frr - far))
    EER = fpr[np.nanargmin(np.absolute((frr - fpr)))]
    EER = (frr[eer_index] + far[eer_index]) / 2
    stats = get_eer_stats(genuine_scores, imposter_scores)
    eer_threshold = stats.eer_th
    EER = stats.eer

    #report = generate_eer_report(stats, ids=[], save_file=Path(output) / "eer_report.txt")

    plt.figure()
    plt.plot(thresholds, frr, label="False Rejection Rate", color='red')
    plt.plot(thresholds, far, label="False Acceptance Rate", color='blue')
    plt.scatter(thresholds[eer_index], EER, color='black')
    plt.text(thresholds[eer_index], EER, f'EER={EER * 100:.1f}%', verticalalignment='bottom',
             horizontalalignment='right')

    plt.xlabel('Decision threshold')
    plt.ylabel('Rate of imposters/genuine comparisons')
    plt.title('FMR, FNMR, and EER')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(Path(output) / f"{label}_EER.png")
    plt.close()

    print(f"EER: {EER * 100:.2f}% at threshold: {eer_threshold}")

    with open(Path(output)/f"{label}_text.txt", "w") as f:
        f.write(f"EER: {EER * 100:.2f}% at threshold: {eer_threshold}")

def parse_list(pairs_path):
    parsed = []
    with open(pairs_path, 'r') as file:
        for line in file:
            path_1, path_2, value = line.strip().split()
            value = bool(int(value))
            parsed.append((path_1, path_2, value))

    return parsed


if __name__ == '__main__':

    parser = ArgumentParser(description="Generate genuine and imposter pairs from a dataset.")
    parser.add_argument("--genuine_path", type=Path, default="",help="Path to the genunine list")
    parser.add_argument("--imposter_path", type=Path, default="",help="Path to the imposter list")
    parser.add_argument("--scores", type=Path, default="", help="saved scores")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output", type=Path, default=".")
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    do_it(args.model, args.genuine_path, args.imposter_path, args.output, args.scores, args.label)
