import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Parameters
data_folder = './data/output'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = './BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = './data/output/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    #bleu1_list=list()
    #bleu2_list=list()
    #bleu3_list=list()
    #bleu4_list=list()
    #result=[]

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        #print(i)
        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

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
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
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
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
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
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)
        #print(allcaps[0])
        #break

        # Hypotheses
        hypotheses.append([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        
        #result_dict={}
        #result_dict['num']=i
        #result_dict['reference']=img_captions
        #result_dict['hypotheses']=[rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        #result.append(result_dict)
        assert len(references) == len(hypotheses)
        #bleu1 = sentence_bleu(img_captions,[w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],weights=(1, 0, 0, 0))
        #bleu2 = sentence_bleu(img_captions,[w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],weights=(0.5, 0.5, 0, 0))
        #bleu3 = sentence_bleu(img_captions,[w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],weights=(0.33, 0.33, 0.33, 0))
        #bleu4 = sentence_bleu(img_captions,[w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],weights=(0.25, 0.25, 0.25, 0.25))
        #print(bleu1)
        #bleu1_list.append(bleu1)
        #bleu2_list.append(bleu2)
        #bleu3_list.append(bleu3)
        #bleu4_list.append(bleu4)
        #print(len(bleu_list))
        #print(img_captions)
        #print([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]) 
        
        
    
    #bleu_dict={}
    #bleu_dict['bleu1']=bleu1_list
    #bleu_dict['bleu2']=bleu2_list
    #bleu_dict['bleu3']=bleu3_list
    #bleu_dict['bleu4']=bleu4_list
    #print(len(references))
    #print(len(hypotheses))
    #with open('./result_hypothese.json','w') as f:
    #   json.dump(result,f)
    #=========matplotlib============
    #print(len(bleu_list))
    #print(bleu_list)
    #plt.hist(bleu_list,bins=20,normed=0,facecolor='blue',edgecolor='black')
    #plt.bar(range(len(bleu_list)),bleu_list,fc='b')
    #plt.plot(range(len(bleu_list)),bleu_list)
    #plt.xlim((0,5000))
    #plt.ylim((0,1))
    #plt.xticks(np.arange(0,5000,1000))
    #plt.yticks(np.arange(0,1,0.1))
    #plt.xlabel('image')
    #plt.ylabel('bleu 1 scores')
    #plt.title('bleu 1')
    #plt.show()
    #sns.distplot(bleu1_list)
    #plt.show()



    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses,weights=(0.25,0.25,0.25,0.25),emulate_multibleu=True)
    bleu3 = corpus_bleu(references, hypotheses,weights=(0.33,0.33,0.33,0),emulate_multibleu=True)
    bleu2 = corpus_bleu(references, hypotheses,weights=(0.5, 0.5, 0, 0),emulate_multibleu=True)
    bleu1 = corpus_bleu(references, hypotheses,weights=(1, 0, 0, 0),emulate_multibleu=True)
    return bleu4,bleu3,bleu2,bleu1


if __name__ == '__main__':
    result_list=[]
    for i in range(1,8):
        beam_size = i
        b4,b3,b2,b1=evaluate(beam_size)
        result={}
        result['beam size']=i
        result['bleu1']=b1
        result['bleu2']=b2
        result['bleu3']=b3
        result['bleu4']=b4
        result_list.append(result)
    with open('./beam_size_result.json','w') as f:
        json.dump(result_list,f)
    #evaluate(beam_size)
    #print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, b4))
    #print("\nBLEU-3 score @ beam size of %d is %.4f." % (beam_size, b3))
    #print("\nBLEU-2 score @ beam size of %d is %.4f." % (beam_size, b2))
    #print("\nBLEU-1 score @ beam size of %d is %.4f." % (beam_size, b1))
