from __future__ import annotations
"""
Event Encoder — War Embedding Pipeline.

Converts raw GDELT/ACLED event descriptions into dense vector embeddings,
then maps them to sector impact scores via the sensitivity matrix.

Uses FinBERT for financial text embeddings with fallback to
sentence-transformers/all-MiniLM-L6-v2 for lighter deployments.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import EVENT_ENCODER_MODEL, EVENT_EMB_DIM
from utils.logger import get_logger, log_step

logger = get_logger('features.event_encoder')


class EventEncoder:
    """
    Neural event encoder using FinBERT or sentence-transformers.
    Classifies event headlines into war event types and computes
    sector impact vectors.
    """

    def __init__(self, model_name: str = EVENT_ENCODER_MODEL):
        """
        Initialize the event encoder.
        
        Args:
            model_name: HuggingFace model name/path
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._use_fallback = False

    def _init_model(self):
        """Lazy initialization of the model."""
        if self._initialized:
            return

        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            logger.info(f'Loading model: {self.model_name}')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self._initialized = True
            logger.info('Model loaded successfully')

        except ImportError:
            logger.warning('transformers not installed, using fallback embeddings')
            self._use_fallback = True
            self._initialized = True

        except Exception as e:
            logger.warning(f'Model loading failed ({e}), using fallback embeddings')
            self._use_fallback = True
            self._initialized = True

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts into dense vector embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of shape (N, 768)
        """
        self._init_model()

        if self._use_fallback:
            return self._fallback_encode(texts)

        import torch

        tokens = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors='pt'
        )

        with torch.no_grad():
            out = self.model(**tokens)

        # Mean pool over token dimension
        embs = out.last_hidden_state.mean(dim=1).numpy()
        return embs

    def _fallback_encode(self, texts: list[str]) -> np.ndarray:
        """
        Fallback encoding using TF-IDF style keyword features.
        Returns (N, 768) dimensional pseudo-embeddings.
        """
        np.random.seed(hash(texts[0]) % 2**31 if texts else 42)

        keywords = [
            'oil', 'gas', 'energy', 'military', 'strike', 'attack',
            'missile', 'war', 'conflict', 'bomb', 'shipping', 'trade',
            'sanctions', 'nuclear', 'ceasefire', 'peace', 'tension',
            'escalation', 'crisis', 'humanitarian', 'cyber', 'hack',
            'houthi', 'hezbollah', 'iran', 'israel', 'gaza', 'lebanon',
            'opec', 'diplomatic', 'technology', 'finance', 'health',
        ]

        embs = []
        for text in texts:
            text_lower = text.lower() if isinstance(text, str) else ''
            # Keyword frequency features
            kw_feats = [text_lower.count(kw) for kw in keywords]
            kw_arr = np.array(kw_feats, dtype=np.float32)

            # Extend to 768 dimensions with reproducible noise
            seed = hash(text) % 2**31 if text else 0
            rng = np.random.RandomState(seed)
            noise = rng.randn(EVENT_EMB_DIM - len(keywords)) * 0.01
            emb = np.concatenate([kw_arr / (np.linalg.norm(kw_arr) + 1e-8), noise])
            embs.append(emb)

        return np.stack(embs)

    def classify_event_type(self, text: str,
                            event_types: list[str]) -> tuple[str, float]:
        """
        Classify a raw headline into one of the 15 war event types
        using cosine similarity between headline embedding and
        event-type label embeddings.
        
        Args:
            text: Raw headline or event description
            event_types: List of event type label strings
            
        Returns:
            Tuple of (best matching event type, confidence score)
        """
        headline_emb = self.encode([text])
        # Create readable labels for event types
        label_texts = [et.replace('_', ' ') for et in event_types]
        label_embs = self.encode(label_texts)

        sims = cosine_similarity(headline_emb, label_embs)[0]
        best_idx = np.argmax(sims)

        return event_types[best_idx], float(sims[best_idx])

    def compute_sector_impact(self, text: str,
                              sensitivity_df: pd.DataFrame) -> pd.Series:
        """
        Full pipeline: text → event type → sector impact vector.
        
        Args:
            text: Raw headline/event description
            sensitivity_df: Sensitivity matrix DataFrame
            
        Returns:
            pd.Series indexed by sector ETF tickers
        """
        ev_type, confidence = self.classify_event_type(
            text, list(sensitivity_df.index)
        )
        impact = sensitivity_df.loc[ev_type] * confidence
        return impact

    def batch_compute_impacts(self, texts: list[str],
                              sensitivity_df: pd.DataFrame) -> list[pd.Series]:
        """Compute sector impacts for a batch of texts."""
        return [self.compute_sector_impact(t, sensitivity_df) for t in texts]


def aggregate_monthly_impacts(events_df: pd.DataFrame,
                               encoder: EventEncoder,
                               sensitivity_df: pd.DataFrame) -> dict:
    """
    Aggregate daily event impacts into monthly snapshots.

    Uses vectorised batch encoding (one FinBERT pass for all events) rather
    than a per-event loop to keep the runtime tractable on CPU.
    """
    log_step(logger, 'Aggregating monthly event impacts',
             f'Events: {len(events_df)}')

    events_df = events_df.copy()
    events_df['month'] = events_df['event_date'].dt.to_period('M')

    # 1) Pre-encode the 15 event-type label strings ONCE
    event_types = list(sensitivity_df.index)
    label_texts  = [et.replace('_', ' ') for et in event_types]
    encoder._init_model()
    label_embs   = encoder.encode(label_texts)                        # (15, 768)
    # L2-normalise for cosine similarity via dot product
    label_norms  = np.linalg.norm(label_embs, axis=1, keepdims=True) + 1e-8
    label_embs_n = label_embs / label_norms                           # (15, 768)

    # 2) Collect all event texts and metadata
    texts, months, fatalities, tones = [], [], [], []
    for _, row in events_df.iterrows():
        text = row.get('title') or row.get('notes', '')
        if not text or not isinstance(text, str):
            continue
        texts.append(text)
        months.append(str(row['month']))
        fatalities.append(float(row.get('fatalities', 0)))
        tones.append(float(row.get('tone_score', 0)))

    if not texts:
        logger.warning('No valid event texts found — returning empty monthly impacts')
        return {}

    logger.info(f'Batch-encoding {len(texts)} event texts …')

    # 3) Batch encode all event texts in chunks to avoid OOM
    CHUNK = 512
    all_embs = []
    for i in range(0, len(texts), CHUNK):
        chunk = texts[i:i + CHUNK]
        all_embs.append(encoder.encode(chunk))
        if (i // CHUNK) % 10 == 0:
            logger.info(f'  … {min(i + CHUNK, len(texts))}/{len(texts)} events encoded')
    event_embs = np.vstack(all_embs)                                  # (N, 768)

    # 4) Cosine similarities: (N, 15)
    event_norms  = np.linalg.norm(event_embs, axis=1, keepdims=True) + 1e-8
    event_embs_n = event_embs / event_norms
    sims         = event_embs_n @ label_embs_n.T                      # (N, 15)

    best_idxs     = np.argmax(sims, axis=1)                           # (N,)
    confidences   = sims[np.arange(len(texts)), best_idxs]            # (N,)

    # 5) Build sensitivity-weighted impact vectors per event: (N, 11)
    sens_matrix = sensitivity_df.values                               # (15, 11)
    impacts_all = sens_matrix[best_idxs] * confidences[:, None]      # (N, 11)
    weights_all = np.array([
        max(np.log1p(f + 1) + abs(t) / 10, 0.1)
        for f, t in zip(fatalities, tones)
    ])                                                                 # (N,)

    # 6) Group by month and compute weighted average
    months_arr = np.array(months)
    unique_months = list(dict.fromkeys(months_arr))                   # preserve order
    monthly = {}
    for m in unique_months:
        mask = months_arr == m
        w    = weights_all[mask]
        imp  = impacts_all[mask]
        monthly[m] = pd.Series(
            np.average(imp, axis=0, weights=w),
            index=sensitivity_df.columns
        )

    logger.info(f'Monthly impacts: {len(monthly)} months computed')
    return monthly


def build_event_embedding_tensor(monthly_impacts: dict,
                                  months: list[str]) -> np.ndarray:
    """
    Build a tensor of monthly event embeddings aligned with graph snapshots.
    
    Args:
        monthly_impacts: Dict from aggregate_monthly_impacts
        months: Ordered list of month strings
        
    Returns:
        numpy array of shape (T, event_emb_dim)
    """
    embs = []
    for m in months:
        if m in monthly_impacts:
            impact = monthly_impacts[m].values
            # Pad to EVENT_EMB_DIM
            padded = np.zeros(EVENT_EMB_DIM)
            padded[:len(impact)] = impact
            embs.append(padded)
        else:
            embs.append(np.zeros(EVENT_EMB_DIM))

    return np.stack(embs)


if __name__ == '__main__':
    encoder = EventEncoder()
    from features.sensitivity_matrix import SENSITIVITY_DF

    test_texts = [
        'Israel launches airstrikes on Gaza',
        'Oil prices surge amid Houthi Red Sea attacks',
        'Ceasefire talks show progress in Middle East',
        'Iran advances nuclear enrichment program',
    ]

    for text in test_texts:
        impact = encoder.compute_sector_impact(text, SENSITIVITY_DF)
        ev_type, conf = encoder.classify_event_type(
            text, list(SENSITIVITY_DF.index))
        print(f'\n"{text}"')
        print(f'  → Event type: {ev_type} (confidence: {conf:.3f})')
        print(f'  → Top 3 sectors: {impact.abs().nlargest(3).to_dict()}')
