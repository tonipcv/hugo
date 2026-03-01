from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    system_prompt: str
    user_template: str
    
    def format(self, **kwargs: Any) -> str:
        return self.user_template.format(**kwargs)


DEFAULT_REWARD_TEMPLATE = PromptTemplate(
    system_prompt=(
        "You are an expert reinforcement learning reward designer. "
        "Your task is to evaluate agent actions and provide a reward signal between -1 and 1. "
        "Consider the state, action taken, and outcome. "
        "Respond with ONLY a single number between -1 and 1, nothing else."
    ),
    user_template=(
        "Environment: {env_name}\n"
        "Previous State: {prev_state}\n"
        "Action Taken: {action}\n"
        "New State: {new_state}\n"
        "Environment Reward: {env_reward}\n"
        "Episode Progress: {progress}%\n\n"
        "Evaluate this transition and provide a reward signal (-1 to 1):"
    )
)

CARTPOLE_TEMPLATE = PromptTemplate(
    system_prompt=(
        "You are evaluating a CartPole balancing task. "
        "The goal is to keep the pole upright by moving the cart left or right. "
        "Reward actions that keep the pole centered and upright. "
        "Penalize actions that let the pole fall or move the cart too far. "
        "Respond with ONLY a number between -1 and 1."
    ),
    user_template=(
        "Cart Position: {cart_pos:.3f}\n"
        "Cart Velocity: {cart_vel:.3f}\n"
        "Pole Angle: {pole_angle:.3f} radians\n"
        "Pole Angular Velocity: {pole_vel:.3f}\n"
        "Action: {'Left' if action == 0 else 'Right'}\n"
        "Steps Survived: {steps}\n\n"
        "Reward (-1 to 1):"
    )
)

TRADING_TEMPLATE = PromptTemplate(
    system_prompt=(
        "You are a professional trading advisor evaluating trading decisions. "
        "Consider risk management, profit potential, and market conditions. "
        "Reward profitable and safe trades, penalize risky or losing trades. "
        "Respond with ONLY a number between -1 and 1."
    ),
    user_template=(
        "Current Price: ${price:.2f}\n"
        "Position: {position}\n"
        "Portfolio Value: ${portfolio_value:.2f}\n"
        "Action: {action}\n"
        "Price Change: {price_change:+.2f}%\n"
        "Profit/Loss: ${pnl:+.2f}\n\n"
        "Evaluate this trade (-1 to 1):"
    )
)

TEMPLATES: Dict[str, PromptTemplate] = {
    "default": DEFAULT_REWARD_TEMPLATE,
    "cartpole": CARTPOLE_TEMPLATE,
    "trading": TRADING_TEMPLATE,
}


def get_default_template() -> PromptTemplate:
    return DEFAULT_REWARD_TEMPLATE


def get_template(name: str) -> PromptTemplate:
    return TEMPLATES.get(name, DEFAULT_REWARD_TEMPLATE)
