import torch
from tqdm import tqdm
from utils import PromptUtils
import random

def get_toolwise_attention_scores(query_span, item_spans, attentions):
    toolwise_scores = torch.zeros((len(item_spans), len(attentions), attentions[0].size(1)), device=attentions[0].device) # shape: (num_tools, num_layers, num_heads)

    for tool_id, tool_span in enumerate(item_spans):
        for layer_ix, layer_attn in enumerate(attentions):
            # layer_attn shape: (batch_size, num_heads, seq_len, seq_len)
            attn_query_to_tool = layer_attn[0, :, query_span[0]:query_span[1], tool_span[0]:tool_span[1]] # shape: (num_heads, query_len, tool_len)
            attn_sum_on_query_tool = attn_query_to_tool.sum(dim=(1, 2)) # shape: (num_heads,)
            toolwise_scores[tool_id, layer_ix] = attn_sum_on_query_tool

    return toolwise_scores

def attention_norm_on_tools_score(toolwise_scores, gold_tool_id):
    score = toolwise_scores[gold_tool_id] / (toolwise_scores.sum(dim=0) + 1e-8) # shape: (num_layers, num_heads)
    return score

def attention_rank_tools_score(toolwise_scores, gold_tool_id):
    # rank tools per head
    sorted_ids = torch.argsort(toolwise_scores, dim=0, descending=True) # shape: (num_tools, num_layers, num_heads)
    gold_tool_ranks = torch.argsort(sorted_ids, dim=0)[gold_tool_id] # shape: (num_layers, num_heads)
    score = 1.0 / (gold_tool_ranks + 1.0) # higher score for better rank
    return score

def select_retrieval_heads(train_queries, model, tokenizer, tools, get_query_span, device, max_heads=20):
    # TODO 3: Head selection
    """
    Identify a subset of attention heads that are most useful for retrieving the correct tool.

    Requirements:
    - Use the same prompt structure as Part-2
    - Use attention patterns(query -> tool) to score heads
    - Aggregate signals across training queries
    - Return "max_heads" heads as (layer, head)

    Notes:
    - You must construct prompts and extract attentions inside this function
    - Avoid hardcoding specific queries or tools
    """

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # accumulate scores per head
    head_scores = torch.zeros(num_layers, num_heads, device=device)

    for qix in tqdm(range(len(train_queries))):

        sample = train_queries[qix]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tool_ids = list(tools.keys())
        random.shuffle(tool_ids)
        putils = PromptUtils(
        tokenizer=tokenizer, 
        doc_ids=tool_ids, 
        dict_all_docs=tools,
        )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)

        # print("Item spans:", item_spans)
        # print("Document lengths:", doc_lengths)
        # print("Map doc name to id:", map_docname_id)
        # print("Map id to doc name:", map_id_docname)
        
        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = inputs.input_ids[0]

        with torch.no_grad():
            attentions = model(**inputs).attentions # list of (batch_size, num_heads, seq_len, seq_len) for each layer

        # Add your head scoring logic after this line
        # gold_tool_span = item_spans[map_docname_id[gold_tool_name]]
        query_span = get_query_span(question, putils, tokenizer)
        gold_tool_id = map_docname_id[gold_tool_name]

        toolwise_scores = get_toolwise_attention_scores(query_span, item_spans, attentions) # shape: (num_tools, num_layers, num_heads)
        head_scores_query_to_tools = attention_norm_on_tools_score(toolwise_scores, gold_tool_id) # shape: (num_layers, num_heads)
        # head_scores_query_to_tools = attention_rank_tools_score(toolwise_scores, gold_tool_id) # shape: (num_layers, num_heads)
        head_scores += head_scores_query_to_tools

    # TODO: select top heads
    selected_heads = []

    values, indices = torch.topk(head_scores.view(-1), k=max_heads)
    for idx in indices:
        layer_ix = (idx // num_heads).item()
        head_ix = (idx % num_heads).item()
        selected_heads.append((layer_ix, head_ix))

    # example expected format:
    # [(layer1, head3), (layer5, head10), ...]
    assert len(selected_heads) == max_heads
    return selected_heads