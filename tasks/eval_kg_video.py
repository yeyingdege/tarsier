import os, sys
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
sys.path.append(os.getcwd())
from tasks.decord_func import decord_video_given_start_end_seconds
from tasks.eval_utils import parse_choice, TypeAccuracy
from tasks.utils import load_model_and_processor



QUESTION_TYPES = ['qa1_step2tool', 'qa2_bestNextStep', 'qa3_nextStep',
                  'qa4_step','qa5_task', 'qa6_precedingStep', 'qa7_bestPrecedingStep',
                  'qa8_toolNextStep', 'qa9_bestInitial','qa10_bestFinal', 'qa11_domain',
                  'qa12_toolPurpose', 'qa13_actionPurpose', 'qa14_objectPurpose',
                  'qa15_ToolOtherPurpose', 'qa16_ObjectOtherPurpose', 'qa17_AlternativeTool',
                  'qa18_AlternativeObject', 'qa19_TaskSameToolSamePurpose',
                  'qa20_TaskSameObjectSamePurpose']



def load_video(video_path, num_segments=8, start_secs=-1, end_secs=-1, return_msg=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

    frame_indices = decord_video_given_start_end_seconds(video_path, 
                        start_secs=start_secs, end_secs=end_secs,
                        num_video_frames=num_segments)

    frames = vr.get_batch(frame_indices).asnumpy()
    # print(frames.shape)
    if frames.shape[0] != num_segments:
        print('Concat frames...')
        frames = torch.from_numpy(frames)
        num_concat_frames = max(num_segments - frames.shape[0], 0)
        concat_frames = torch.zeros((num_concat_frames, frames.shape[1], frames.shape[2], frames.shape[3])).type_as(frames).to(frames.device)
        frames = torch.cat([frames, concat_frames], dim=0).numpy()

    frames = [Image.fromarray(v.astype('uint8')) for v in frames]

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames


def process_one(model, processor, prompt, video_file, generate_kwargs,
                num_frame=8, start_secs=-1, end_secs=-1):
    frames, msg = load_video(
        video_file, num_segments=num_frame, return_msg=True, 
        start_secs=start_secs, end_secs=end_secs
    )
    # print('start_secs', start_secs, 'end_secs', end_secs, msg)
    inputs = processor(prompt, images=frames, edit_prompt=True, return_prompt=True)
    if 'prompt' in inputs:
        # print(f"Prompt: {inputs.pop('prompt')}")
        inputs.pop('prompt')
    inputs = {k:v.to(model.device) for k,v in inputs.items() if v is not None}
    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )
    output_text = processor.tokenizer.decode(outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True)
    return output_text


def main(args):
    model, processor = load_model_and_processor(args.model_path, max_n_frames=args.num_video_frames)
    generate_kwargs = {
        "do_sample": True if args.temperature > 0 else False,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "use_cache": True
    }

    # Load Questions
    annotations = json.load(open(os.path.expanduser(args.question_file), "r"))

    # Overall Accuracy for All Questions
    global_acc = TypeAccuracy("Global")
    qa_acc = []
    for t in range(len(QUESTION_TYPES)):
        qa_acc.append(TypeAccuracy(f"qa{t+1}_"))


    total = 0
    results = {}
    for line in tqdm(annotations, total=len(annotations)):
        # Q-A Pair
        idx = line["qid"]
        quest_type = line["quest_type"]
        conversations = line["conversations"]
        qs = conversations[0]["value"]
        gt_answers = conversations[1]["value"]
        results[idx] = {"qid": idx, "quest_type": quest_type, 
                        "qs": qs, "gt": gt_answers,
                        "task_label": line["task_label"], 
                        "step_label": line["step_label"]}
        qs = qs.replace("<video>\n", "").replace("<image>\n", "")
        vid_path = os.path.join(args.image_folder, line["video"])
        if "start_secs" in line:
            start_secs = line['start_secs']
            end_secs = line['end_secs']
        else:
            start_secs = -1
            end_secs = -1
        prompt = '<video>\n' + qs
        response = process_one(model, processor, prompt, vid_path, generate_kwargs,
                               num_frame=args.num_video_frames, start_secs=start_secs, end_secs=end_secs)
        response = response.strip()
        total += 1
        answer_id = parse_choice(response, line["all_choices"], line["index2ans"])
        results[idx]["response"] = response
        results[idx]["parser"] = answer_id
        # print("qid {}:\n{}".format(idx, qs))
        # print("AI: {}\nParser: {}\nGT: {}\n".format(response, answer_id, gt_answers))

        global_acc.update(gt_answers, answer_id)
        for t in range(len(QUESTION_TYPES)):
            if f"qa{t+1}_" in quest_type:
                qa_acc[t].update(gt_answers, answer_id)

        # print each type accuracy
        print("-----"*5)
        acc_list = []
        for t in range(len(QUESTION_TYPES)):
            qa_acc[t].print_accuracy()
            acc_list.append(qa_acc[t].get_accuracy())
        global_acc.print_accuracy()
        print("-----"*5)
        avg_acc = sum(acc_list) / len(acc_list)
        print("Average Acc over Type: {:.4f}".format(avg_acc))

    # save all results
    print("save to {}".format(args.answers_file))
    with open(args.answers_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Process Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="omni-research/Tarsier-7b")
    parser.add_argument("--image-folder", type=str, default="data/COIN/videos")
    parser.add_argument("--question-file", type=str, default="data/testing_vqa20.json")
    parser.add_argument("--answers-file", type=str, default="data/answers_tarsier7b_f8.json")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_video_frames", type=int, default=8)
    args = parser.parse_args()
    main(args)

