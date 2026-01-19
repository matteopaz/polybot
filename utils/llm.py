from openrouter import OpenRouter
from dotenv import load_dotenv
import os
import concurrent.futures
import time

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

openrouter_client = OpenRouter(api_key=OPENROUTER_API_KEY)

def insider_event_score(event_title: str, max_retries: int = 3) -> int:
    system_prompt = """
    [ROLE]
    You are an expert market analyst, specializing in prediction markets such as Polymarket.
    Your task is to evaluate the "insider score" of a given event, essentially the degree to which it is
    possible that insiders could be participating in the market, based on the nature of the event and its context.

    The insider score should be the product of several components:
    1. The relative quantity, 1 to 5, of individuals likely to posess some non-public information about the event. 1 being a handful, 5 being entire companies or huge groups.
    2. The relative advantage, 1 to 5, that non-public information would confer in predicting the outcome of the event. The more stochastic the event, the lower this must be.
    3. The relative incentive, 1 to 5, for individuals with non-public information to participate in the market. Financial incentives are always present obviously, but it would be unlikely for insiders to bet if they are already heavily dependent on the event in other ways. 

    Be pragmatic, think about exactly how practical it might be to use inside information, and how motivated insiders would be to do so.

    [EXAMPLES]

    Event: "Chicago Bulls vs Los Angeles Lakers"
    Analysis:
    1. Relative Quantity: 2 (Only the players, management, and close affiliates would have non-public information or insights.)
    2. Relative Advantage: 1 (Assuming the game is not fixed, non-public information would have limited impact on the outcome.)
    3. Relative Incentive: 1 (Insiders have some additional financial incentive to bet, but they are already likely financially invested in the team's success.)
    Insider Score = 2

    Event: "Will NYC have more than 2 inches of snow on December 25, 2025?"
    Analysis:
    1. Relative Quantity: 1 (Very few individuals have access to non-public information, perhaps only certain meteorologists with specialized data.)
    2. Relative Advantage: 1 (Weather predictions are largely based on public data, and are extremely stochastic.)
    3. Relative Incentive: 3 (By betting on a prediction market, insiders could potentially profit from their specialized knowledge in a way that they cannot otherwise.)
    Insider Score = 3

    Event: "Will OpenAI release GPT-5 by the end of 2025?"
    Analysis:
    1. Relative Quantity: 5 (Numerous employees, contractors, and close affiliates likely have non-public information about the development and release plans.)
    2. Relative Advantage: 5 (The non-public information would be definitive in predicting the outcome of the event.)
    3. Relative Incentive: 5 (Insiders have a strong financial incentive to bet on the market, as they may stand to gain significantly from accurate predictions.)
    Insider Score = 125

    Event: "Presidential Election Winner 2028"
    Analysis:
    1. Relative Quantity: 5 (Numerous campaign staff, lobbyists, close affiliates -- elections are massive operations)
    2. Relative Advantage: 2 (While insiders may have some good information, most of the important information is public. In addition, it is unlikely that insiders exist which have all the necessary information to predict the outcome with high certainty.)
    3. Relative Incentive: 2 (Insiders are likely already financially invested in the outcome of the election through donations and other means, reducing their incentive to bet on the market.)
    Insider Score = 20

    [INSTRUCTIONS]
    You will be given the title of an event. Please conduct a multifaceted and detailed analysis to determine the insider score, following the structure and types of reasoning demonstrated in the examples above.

    Your message should ALWAYS end with a line that states "Score: X", where X is the computed insider score. Do not include any other text after this line or any other text on this line.
    """

    user_prompt = f"Evaluate this event: \"{event_title}\""

    response = openrouter_client.chat.send(
        model="google/gemini-2.5-flash-lite",
        messages=[
            {
                "role": "system", 
                "content": [
                    dict(type="text", 
                         text=system_prompt,
                         cache_control={"type": "ephemeral", "ttl": "1h"}
                        ),
                ]
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.5
    )

    text = response.choices[0].message.content.strip().lower()
    for attempt in range(max_retries):
        try:
            for line in text.splitlines():
                if line.startswith("score:"):
                    score_str = line.split("score:")[1].strip()
                    score = int(score_str)
                    return score
            raise ValueError("Score not found in response.")
        except Exception as e:
            if attempt < max_retries - 1:
                continue

    print("Warning: Failed to parse score from LLM response for event:", event_title)
    return 0

def insider_event_score_parallel(event_titles: list[str]) -> list[int]:
    scores = [0 for _ in event_titles]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(insider_event_score, title): idx
            for idx, title in enumerate(event_titles)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                score = future.result()
                scores[idx] = score
            except Exception as e:
                print(f"Error computing score for '{event_titles[idx]}': {e}")
                scores[idx] = 0
    return scores
