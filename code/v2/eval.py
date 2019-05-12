import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
#from nlgeval import NLGEval

from tqdm import tqdm

# Parameters
data_folder = './output/'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = './BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = './output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

#nlgeval = NLGEval()
# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST'),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (img_att,img_fc, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        img_att = img_att.to(device)
        img_fc = img_fc.to(device)
        image_features = torch.cat([img_att,img_fc],dim=1) # (batch_size, num_pixel,features_dim)
        
        features_dim = image_features.size(-1) # 2048

        num_pixels = image_features.size(1)  # 37
        
        # We'll treat the problem as having a batch size of k
        #image_features = image_features.expand(k, num_pixels, features_dim)  # (k, num_pixels, encoder_dim)

        img_fc = img_fc.expand(k,1,2048)
        img_fc = img_fc.view(k,2048)
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1 = decoder.init_hidden_state(k)
        h2, c2 = decoder.init_hidden_state(k)
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            #print(img_fc.size())
            h1,c1 = decoder.language_lstm(torch.cat([embeddings,img_fc.squeeze(1),h2],dim=1),(h1,c1))

            awe, _ = decoder.attention(image_features,h1,h2)  # (s, encoder_dim), (s, num_pixels)

            #gate = decoder.sigmoid(decoder.f_beta(h1))  # gating scalar, (s, encoder_dim)
            #awe = gate * awe

            h2, c2 = decoder.attention_lstm(awe, (h2, c2))  # (s, decoder_dim)

            scores = decoder.output(h1,h2)


            #scores = decoder.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            #image_features = image_features[prev_word_inds[incomplete_inds]]
            img_fc = img_fc[prev_word_inds[incomplete_inds]]

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads

        #img_caps = [' '.join(c) for c in img_captions ]
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        #hypothesis=' '.join(hypothesis)
        #hypotheses.append(hypothesis)
        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses,weights=(0.25,0.25,0.25,0.25),emulate_multibleu=True)
    bleu3 = corpus_bleu(references, hypotheses,weights=(0.33,0.33,0.33,0),emulate_multibleu=True)
    bleu2 = corpus_bleu(references, hypotheses,weights=(0.5, 0.5, 0, 0),emulate_multibleu=True)
    bleu1 = corpus_bleu(references, hypotheses,weights=(1, 0, 0, 0),emulate_multibleu=True)
    
    #calulate scores
    #metrics_dict = nlgeval.compute_metrics(references,hypotheses)   
    return bleu4,bleu3,bleu2,bleu1


if __name__ == '__main__':
    beam_size = 3
    b4,b3,b2,b1=evaluate(beam_size)
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, b4))
    print("\nBLEU-3 score @ beam size of %d is %.4f." % (beam_size, b3))
    print("\nBLEU-2 score @ beam size of %d is %.4f." % (beam_size, b2))
    print("\nBLEU-1 score @ beam size of %d is %.4f." % (beam_size, b1))
    print('\n*********************\n')
    #print(metrics_dict)
