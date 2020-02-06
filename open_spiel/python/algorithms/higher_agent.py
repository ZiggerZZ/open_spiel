# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RL agent that always chooses the next higher possible legal action
over opponents actions if partner's action isn't higher than opponents' auctions;
otherwise plays the lowest.
Only for NT contracts in bridge."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from open_spiel.python import rl_agent

def beat_or_play_lowest(cur_legal_actions, opponent_card):
  suit = opponent_card % 4
  # if at least one legal card is of the same suit, then all of them are
  if cur_legal_actions[0] % 4 == suit:
    # cur_legal_actions should be sorted
    for card in cur_legal_actions:
      if card > opponent_card:
        return card
  # if not the same suit or no card higher found
  return np.min(cur_legal_actions)

def permute_players(current_trick_us, current_trick_lh, current_trick_pd, current_trick_rh):
  return current_trick_pd, current_trick_rh, current_trick_us, current_trick_lh


class HigherAgent(rl_agent.AbstractAgent):
  """Higher agent class."""

  def __init__(self, player_id, num_actions, name="higher_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    # Pick the higher or the minimal legal action.
    cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
    minimal_action = np.min(cur_legal_actions)
    info_state = time_step.observations["info_state"][self._player_id]
    # if lead
    if info_state[1] == 1.0:
      action = minimal_action
      probs = np.zeros(self._num_actions)
      probs[action] = 1.0
      return rl_agent.StepOutput(action=action, probs=probs)
    # 337:337+52*4 - indices for current_trick observation
    current_trick = info_state[337:337+52*4]
    # order of players in current_trick: Us, LH, Pd, RH
    # careful! for dummy all is permuted: Us -> Pd, Pd -> Us, Lh -> Rh, Rh -> Lh
    # "Us", "LH", "Pd", "RH"
    # DECLARER
    # us = 0 everywhere
    # lh = 0, pd = 0, rh = 0 -> we are first 0 0 0 0
    # lh = 0, pd = 0, rh = 1 -> we are second 0 0 0 1
    # lh = 0, pd = 1, rh = 1 -> we are third 0 0 1 1
    # lh = 1, pd = 1, rh = 1 -> we are fourth 0 1 1 1
    # DUMMY
    # pd = 0 everywhere
    # lh = 0, us = 0, rh = 0 -> we are first 0 0 0 0
    # lh = 1, us = 0, rh = 0 -> we are second 0 1 0 0
    # lh = 1, us = 1, rh = 0 -> we are third 1 1 0 0
    # lh = 1, us = 1, rh = 1 -> we are fourth 1 1 0 1
    current_trick_us = current_trick[:52]
    current_trick_lh = current_trick[52:104]
    current_trick_pd = current_trick[104:156]
    current_trick_rh = current_trick[156:208]
    played_us = bool(np.sum(current_trick_us))
    played_lh = bool(np.sum(current_trick_lh))
    played_pd = bool(np.sum(current_trick_pd))
    played_rh = bool(np.sum(current_trick_rh))
    bools = [played_us, played_lh, played_pd, played_rh]
    first = [0, 0, 0, 0]
    declarer = [[0,0,0,1],[0,0,1,1],[0,1,1,1]]
    dummy = [[0,1,0,0],[1,1,0,0],[1,1,0,1]]
    if bools == first:
      action = minimal_action
    for i in range(3):
      if bools == dummy[i]:
        current_trick_us, current_trick_lh, current_trick_pd, current_trick_rh = \
        permute_players(current_trick_us, current_trick_lh, current_trick_pd, current_trick_rh)
        played_us = bool(np.sum(current_trick_us))
        played_lh = bool(np.sum(current_trick_lh))
        played_pd = bool(np.sum(current_trick_pd))
        played_rh = bool(np.sum(current_trick_rh))
        bools = [played_us, played_lh, played_pd, played_rh]
      if bools == declarer[i]:
        # second seat
        if i == 0:
          card_rh = np.argmax(current_trick_rh)
          action = beat_or_play_lowest(cur_legal_actions, card_rh)
        # third seat
        if i == 1:
          card_pd = np.argmax(current_trick_pd)
          suit_pd = card_pd % 4
          card_rh = np.argmax(current_trick_rh)
          suit_rh = card_rh % 4
          if suit_pd != suit_rh:
            # opponent didn't follow the suit so partner's card is the highest
            action = minimal_action
          else:
            if card_pd > card_rh:
              # partner's card is the highest
              action = minimal_action
            else:
              # check if I can beat and beat if I can
              action = beat_or_play_lowest(cur_legal_actions, card_rh)
        # fourth seat
        if i == 2:
          card_lh = np.argmax(current_trick_lh)
          suit_lh = card_lh % 4
          card_pd = np.argmax(current_trick_pd)
          suit_pd = card_pd % 4
          card_rh = np.argmax(current_trick_rh)
          suit_rh = card_rh % 4
          if suit_pd == suit_lh == suit_rh:
            if card_pd > max(card_lh, card_rh):
              action = minimal_action
            else:
              action = beat_or_play_lowest(cur_legal_actions, max(card_lh, card_rh))
          elif suit_pd == suit_lh:
            if card_pd > card_lh:
              action = minimal_action
            else:
              action = beat_or_play_lowest(cur_legal_actions, card_lh)
          elif suit_lh == suit_rh:
            action = beat_or_play_lowest(cur_legal_actions, max(card_lh, card_rh))
          else:
            action = beat_or_play_lowest(cur_legal_actions, card_lh)

    probs = np.zeros(self._num_actions)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)