import re
from collections import OrderedDict

class InfoCard():
    def __init__(self, tokenizer=None, width=None, shift_vocab_num=None, title=None):
        self.tokenizer = tokenizer
        self.width = width or 100
        self.shift_vocab_num = shift_vocab_num or 0
        self.title = title or " INFO CARD "
        self._init_card()

    def _init_card(self):
        init_info = ["#"* self.width, f"{self.title:{'#'}^{self.width}}", "#"* self.width]
        self.info = "\n" + "\n".join(init_info) + "\n"

    def add_title_line(self, title=None):
        if title:
            self.info += f"{title:{'='}^{self.width}}\n"

    def pad_line(self, line):
        padded_line = line.ljust(self.width)
        return padded_line + '\n'

    def token_to_rank(self, emb):
        sorted_index = emb.topk(emb.shape[0]).indices.detach().cpu().numpy().tolist()        
        sorted_index = [int(x)+self.shift_vocab_num for x in sorted_index]
        sorted_token = self.tokenizer.convert_ids_to_tokens(sorted_index)
        token_to_rank = OrderedDict({token:rank for rank, token in enumerate(sorted_token)})
        return token_to_rank

    def add_stat_info(self, embs, title=None):
        self.add_title_line(title)
        N, V = embs.shape
        f_mean = (embs.sum(-1) / (embs!=0).sum(-1)).mean()
        f_max = embs.max(-1)[0].mean()
        f_min = embs.min(-1)[0].mean()
        f_nonzero = embs.bool().sum(-1).float().mean()
        slot_shape = f"shape: ({N}, {V})"
        slot_nonzero = f" gate: {f_nonzero:>5.1f}/{V:<5.0f}={f_nonzero/V*100:<4.2f}% "
        slot_mean = f" mean: {f_mean:<7.2f} "
        slot_max = f" max: {f_max:<7.2f} "
        slot_min = f" min: {f_min:<7.2f} "
        slot_width = [int(x * self.width) for x in [0.20, 0.30, 0.15, 0.15, 0.14]]
        stat_info = f"{slot_shape:^{slot_width[0]}}|{slot_nonzero:^{slot_width[1]}}|{slot_mean:^{slot_width[2]}}|{slot_max:^{slot_width[3]}}|{slot_min:^{slot_width[4]}}"        
        self.info += stat_info + "\n"

    def add_text_info(self, text, title=None):
        text_items = re.split(r'(\s+)', text)
        text = self.tidy_item(text_items)
        title = title or " TEXT "
        title = f"{title:{'='}^{self.width}}"
        self.info += f"{title}\n{text}\n"

    def add_texts_info(self, texts, title=None, descs=None):
        self.add_title_line(title)
        assert descs is None or len(texts) == len(descs)
        for i, text in enumerate(texts):
            if descs:
                text = f"{descs[i]}: {text}"
                text_items = re.split(r'(\s+)', text)
                self.info += self.tidy_item(text_items) + "\n\n"
            else:
                text_items = re.split(r'(\s+)', text)
                self.info += self.tidy_item(text_items) + "\n\n"

    def add_interaction_info(self, q_emb, p_emb, p2_emb=None, k=20, title=None):
        self.add_title_line(title)

        token_to_rank_q = self.token_to_rank(q_emb)
        token_to_rank_p = self.token_to_rank(p_emb)
        token_to_rank_qp = self.token_to_rank(q_emb * p_emb)

        token_to_rank2_q = [(t, token_to_rank_q[t], token_to_rank_p[t]) for t, _ in token_to_rank_q.items()][:k]
        self.add_title_line(' V(q) => (t, qrank, prank)')
        self.info += self.tidy_item(token_to_rank2_q) + "\n"
        
        token_to_rank2_p = [(t, token_to_rank_q[t], token_to_rank_p[t]) for t, _ in token_to_rank_p.items()][:k]
        self.add_title_line(' V(p) => (t, qrank, prank)')
        self.info += self.tidy_item(token_to_rank2_p) + "\n"

        if p2_emb is not None:
            token_to_rank_p_neg = self.token_to_rank(p2_emb)
            token_to_rank2_p_neg = [(t, token_to_rank_q[t], token_to_rank_p_neg[t]) for t, _ in token_to_rank_p_neg.items()][:k]
            self.add_title_line(' V(p_neg) => (t, qrank, pnegrank) ')
            self.info += self.tidy_item(token_to_rank2_p_neg) + "\n"

        token_to_rank2_qp = [(t, token_to_rank_q[t], token_to_rank_p[t]) for t, _ in token_to_rank_qp.items()][:k]
        self.add_title_line(' V(q) * V(p) => (t, qrank, prank)')
        self.info += self.tidy_item(token_to_rank2_qp) + "\n"


    def add_example(self, q_text, q_emb, p_text, p_emb, p_neg_text=None, p_neg_emb=None):
        title = " EXAMPLE "
        title = f"{title:{'='}^{self.width}}"
        q_text = self.tidy_item(f"[Q_TEXT]: {q_text}".split())
        p_text = self.tidy_item(f"[P_TEXT]: {p_text}".split())
        if p_neg_text is not None:
            p_neg_text = self.tidy_item(f"[P_TEXT]: {p_neg_text}".split())
            self.info += f"{title}\n{q_text}\n{p_text}\n{p_neg_text}\n"
        else:
            self.info += f"{title}\n{q_text}\n{p_text}\n"
        
        token2qrank = self.get_token_to_rank(q_emb)
        token2prank = self.get_token_to_rank(p_emb)
        token2simrank = self.get_token_to_rank(q_emb * p_emb)

        k = 20
        q_topk_tokens = [(t, rank, token2prank[t]) for t,rank in token2qrank.items()][:k]
        disentangle_info = self.tidy_item(q_topk_tokens)
        q_emb_name = ' V(q) '
        disentangle_title = f"{q_emb_name :{'='}^{self.width}}"
        self.info += f"{disentangle_title}\n{disentangle_info}" + "\n"
        
        p_topk_tokens = [(t, token2qrank[t], rank) for t,rank in token2prank.items()][:k]
        disentangle_info = self.tidy_item(p_topk_tokens)
        p_emb_name = ' V(p) '
        disentangle_title = f"{p_emb_name :{'='}^{self.width}}"
        self.info += f"{disentangle_title}\n{disentangle_info}" + "\n"
        
        if p_neg_text is not None and p_neg_emb is not None:
            token2negrank = self.get_token_to_rank(p_neg_emb)
            p_neg_topk_tokens = [(t, token2qrank[t], rank) for t,rank in token2negrank.items()][:k]
            disentangle_info = self.tidy_item(p_neg_topk_tokens)
            p_neg_emb_name = ' V(p_neg) '         
            disentangle_title = f"{p_neg_emb_name :{'='}^{self.width}}"
            self.info += f"{disentangle_title}\n{disentangle_info}" + "\n"
            
        sim_topk_tokens = [(t, token2qrank[t], token2prank[t]) for t,rank in token2simrank.items()][:k]
        disentangle_info = self.tidy_item(sim_topk_tokens)
        title = f' V(q) * V(p) '
        disentangle_title = f"{title :{'='}^{self.width}}"
        self.info += f"{disentangle_title}\n{disentangle_info}" + "\n"



    def tidy_item(self, items):        
        info = ""
        row = ""
        for item in items:
            if item in ["\n", "\n\n"]:
                row = self.pad_line(row)
            elif str(item).isspace():
                pass
            elif len(row) + len(str(item)) < self.width:
                row += f"{item} "
            else:
                info += row + "\n"
                row = f"{item} "
        info += row + "\n"
        info = info.strip()
        return info

    def wrap_info(self):
        lines = ["\n"]
        for line in self.info.strip().split("\n"):
            line_ = f"### {line:<{self.width}} ###"
            lines.append(line_)
        lines.append("#"*len(line_))
        info = "\n".join(lines)
        self.info = info


