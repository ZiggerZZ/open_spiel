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

"""Tests for google3.third_party.open_spiel.python.algorithms.maximal_agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import maximal_agent


class MaximalAgentTest(absltest.TestCase):

  def test_step(self):
    agent = maximal_agent.MaximalAgent(player_id=0, num_actions=10)

    legal_actions = [0, 2, 3, 5]
    time_step = rl_environment.TimeStep(
        observations={
            "info_state": [[0], [1]],
            "legal_actions": [legal_actions, []],
            "current_player": 0
        },
        rewards=None,
        discounts=None,
        step_type=None)
    agent_output = agent.step(time_step)

    self.assertIn(agent_output.action, legal_actions)
    self.assertAlmostEqual(sum(agent_output.probs), 1.0)
    self.assertEqual(agent_output.action, 5)


if __name__ == "__main__":
  absltest.main()
