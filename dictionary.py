q_types_mapping = {
    'abnormality_color': 'color',
    'landmark_color': 'color',
    'abnormality_location': 'location',
    'instrument_location': 'location',
    'landmark_location': 'location',
    'finding_count': 'count',
    'instrument_count': 'count',
    'polyp_count': 'count',
    'abnormality_presence': 'yesno',
    'box_artifact_presence': 'yesno',
    'finding_presence': 'yesno',
    'instrument_presence': 'yesno',
    'landmark_presence': 'yesno',
    'text_presence': 'yesno',
    'polyp_removal_status': 'yesno',
    'polyp_type': 'single',
    'polyp_size': 'single',
    'procedure_type': 'single',
}
q_types = ["yesno", "single", "multi", "color", "location", "count"]

def normalize_answer(ans, q_type):
    ans = ans.strip().lower()

    if q_type == "yesno":
        if "yes" in ans or "present" in ans or "evidence" in ans:
            return "Yes"
        elif "no" in ans or "absent" in ans or "none" in ans:
            return "No"
        else:
            return None  # ambiguous

    if q_type == "count":
        # Extract numeric value or return None
        from re import findall
        numbers = findall(r"\d+", ans)
        if numbers:
            return numbers[0]
        elif "one" in ans: return "1"
        elif "two" in ans: return "2"
        return None

    if q_type == "color":
        for color in ["red","green","yellow","blue","white","black"]:
            if color in ans:
                return color
        return None

    if q_type == "location":
        # Simplify locations to a small fixed set
        for loc in ["upper","lower","left","right","central"]:
            if loc in ans:
                return loc
        return None

    if q_type in ["single","multi"]:
        return ans  # keep original but can also restrict choices

    return ans


def build_vocabs(dataset):
    # Build task-specific vocabularies
    task_vocabs = {}
    for general_class in set(q_types_mapping.values()):
        task_vocabs[general_class] = {}
    
    for row in dataset:
        fine_class = row["question_class"]

        # ✅ Handle if fine_class is a list
        if isinstance(fine_class, list):
            fine_class = fine_class[0]  

        general_class = q_types_mapping[fine_class]

        norm_ans = normalize_answer(row["answer"], general_class)
        if norm_ans is None:
            continue  # skip unnormalizable answers

        if norm_ans not in task_vocabs[general_class]:
            idx = len(task_vocabs[general_class])
            task_vocabs[general_class][norm_ans] = idx

    return task_vocabs

from collections import defaultdict

def build_answer_vocab(dataset, q_types_mapping):
    answer_vocab = defaultdict(dict)
    counters = defaultdict(int)

    for ans, q_class in zip(dataset["answer"], dataset["question_class"]):
        # q_class might be a list; pick the first (if multiple labels)
        if isinstance(q_class, list):
            q_class = q_class[0]

        general_class = q_types_mapping[q_class]

        if ans not in answer_vocab[general_class]:
            answer_vocab[general_class][ans] = counters[general_class]
            counters[general_class] += 1

    return answer_vocab

task_vocabs=build_vocabs(train_data)
answer_vocabs = build_answer_vocab(train_data, q_types_mapping)