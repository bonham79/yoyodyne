"""Encoder classes for feature reliant architectures.
"""

import argparse
from typing import Callable, Dict, Optional, Tuple
import math
import pytorch_lightning as pl
import torch
from torch import nn, optim
from . import base
from .. import batches, defaults, evaluators, schedulers, util


class BaseFeaturesEncoder(pl.LightningModule):
	# Indices.
	pad_idx: int
	start_idx: int
	end_idx: int
	# Sizes.
	features_size: int
	# Regularization arguments.
	dropout: float
	label_smoothing: Optional[float]
	# Model arguments.
	embedding_size: int
	encoder_layers: int
	features_size: int
	hidden_size: int
	init_style: str
	# Constructed inside __init__.
	dropout_layer: nn.Dropout

	def __init__(
		self,
		*,
		pad_idx,
		start_idx,
		end_idx,
		features_size,
		label_smoothing=None,
		embedding_size=defaults.EMBEDDING_SIZE,
		encoder_layers=defaults.ENCODER_LAYERS,
		hidden_size=defaults.HIDDEN_SIZE,
		init_style='normal',
		**kwargs,  # Ignored.
	):
		super().__init__()
		self.pad_idx = pad_idx
		self.start_idx = start_idx
		self.end_idx = end_idx
		dropout=dropout
		label_smoothing=label_smoothing
		self.features_size= features_size
		self.embedding_size = embedding_size
		self.encoder_layers = encoder_layers
		self.hidden_size = hidden_size
		self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)
		# Initializes feature embeddings.
		if init_style == 'xavier':
			self.features_embedding = base.BaseEncoderDecoder._xavier_embedding_initialization(self.features_size, self.embedding_size, self.pad_idx)
		elif init_style == 'normal':
			self.features_embedding = base.BaseEncoderDecoder._normal_embedding_initialization(self.features_size, self.embedding_size, self.pad_idx)
		else:
			raise ValueError("Feature embeddings only supports 'xavier' and 'normal' initializations.")

	def forward(
		self, features: batches.PaddedTensor
	) -> torch.Tensor:
		"""Forward step to encode features.

		Args:
			features (batches.PaddedTensor): source padded tensors and mask
				for source, of shape B x seq_len x 1.

		Raises:
			NotImplementedError: This method needs to be overridden.

		Returns:
			torch.Tensor:
				encoded features.
		"""
		raise NotImplementedError

	def append_features(self, batch: batches.PaddedBatch, encoded: torch.Tensor) -> torch.Tensor:
		"""Appends feature encoding to tensor encoding.

		Averages encoded feature values to provide unit length feature embedding before
			concatenating to source_encoded vector.

		Args:
			batch (batches.PaddedBatch): source padded tensors and mask
				for source, of shape B x seq_len x 1.
			encoded (torch.Tensor): Source encoding of shape
				B x seq_len x embedding_size

		Returns:
			torch.Tensor: Concatenation of source_encoded with average of feature values
				(across length).
		"""
		features_encoded = self.forward(batch.features)
		# Averages features over non-masked inputs.
		features_encoded = torch.sum(features_encoded, dim=1)
		denom = torch.sum(~batch.features.mask, dim=1)
		features_encoded = features_encoded / denom
		# Broadcasts features and applies source mask.
		features_encoded = torch.where(batch.source.mask, self.pad_idx, features_encoded)
		# Concatenates to source_encoded.
		encoder_out_feat = torch.cat((encoded, features_encoded), dim=2)
		return encoder_out_feat


class LSTMFeaturesEncoder(BaseFeaturesEncoder):
	"""LSTM Feature Encoder.
	"""
	# Model arguments.
	bidirectional: bool
	encoder: nn.LSTM
	h0: nn.Parameter
	c0: nn.Parameter
	encoder_size: int

	def __init__(
		self,
		*args,
		bidirectional=defaults.BIDIRECTIONAL,
		**kwargs,
	):
		"""Initializes features encoderr.

		Args:
			*args: passed to superclass.
			bidirectional (bool).
			**kwargs: passed to superclass.
		"""
		super().__init__(*args, **kwargs)
		self.bidirectional = bidirectional
		self.encoder = nn.LSTM(
			self.embedding_size,
			self.hidden_size,
			num_layers=self.encoder_layers,
			dropout=self.dropout,
			batch_first=True,
			bidirectional=self.bidirectional,
		)
		self.encoder_size = self.hidden_size * self.num_directions

	@property
	def num_directions(self):
		return 2 if self.bidirectional else 1

	def forward(
		self, features: batches.PaddedTensor
	) -> torch.Tensor:
		"""Encodes features.

		Args:
			features (batches.PaddedTensor): source padded tensors and mask
				for features, of shape B x seq_len x 1.

		Returns:
			torch.Tensor:
				encoded timesteps.
		"""
		embedded = self.features_embedding(features.padded)
		embedded = self.dropout_layer(embedded)
		# Packs embedded source symbols into a PackedSequence.
		packed = nn.utils.rnn.pack_padded_sequence(
			embedded, features.lengths(), batch_first=True, enforce_sorted=False
		)
		# -> B x seq_len x encoder_dim, (h0, c0).
		packed_outs, (H, C) = self.encoder(packed)
		encoded, _ = nn.utils.rnn.pad_packed_sequence(
			packed_outs,
			batch_first=True,
			padding_value=self.pad_idx,
			total_length=None,
		)
		return encoded


class TransformerFeaturesEncoder(BaseFeaturesEncoder):
	"""Transformer features encoder."""

	# Model arguments.
	attention_heads: int
	# Constructed inside __init__.
	encoder: nn.TransformerEncoder
	encoder_size: int

	def __init__(
		self,
		*args,
		attention_heads=defaults.ATTENTION_HEADS,
		**kwargs,
	):
		"""Initializes the encoder-decoder with attention.

		Args:
			attention_heads (int).
			max_source_length (int).
			*args: passed to superclass.
			**kwargs: passed to superclass.
		"""
		super().__init__(*args, **kwargs)
		self.attention_heads = attention_heads
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=self.embedding_size,
			dim_feedforward=self.hidden_size,
			nhead=self.attention_heads,
			dropout=self.dropout,
			activation="relu",
			norm_first=True,
			batch_first=True,
		)
		self.encoder = nn.TransformerEncoder(
			encoder_layer=encoder_layer,
			num_layers=self.encoder_layers,
			norm=nn.LayerNorm(self.embedding_size),
		)
		self.encoder_size = self.hidden_size

	def forward(self, features: batches.PaddedTensor) -> torch.Tensor:
		"""Encodes the source with the TransformerEncoder.

		Args:
			source (batches.PaddedTensor).

		Returns:
			torch.Tensor: sequence of encoded symbols.
		"""
		embedded = self.features_embedding(features.padded)
		embedded = self.dropout_layer(embedded)
		return self.encoder(embedded, src_key_padding_mask=features.mask)


class LinearFeaturesEncoder(BaseFeaturesEncoder):
	"""Linear feature encoder.

		Wrapper class to return just the linear embeddings.
	"""

	def __init__(
		self,
		*args,
		**kwargs,
	):
		super().__init__(*args, **kwargs)
		self.encoding_size = self.embedding_size

	def forward(self, features: batches.PaddedTensor) -> torch.Tensor:
		"""Encodes source embeddings with a linear projection.

		This is a wrapper for just returning feature embeddings.

		Args:
			features (batches.PaddedTensor).

		Returns:
			torch.Tensor: sequence of encoded symbols.
		"""
		return self.features_embedding(features.padded)

