<p align="center">
  <img src="extra/project_logo.png" width="200" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">LEMSS Platform</h1>
</p>
<p align="center">
    <em><code>► LLM based Ecosystem for Multi-agent competitive Search Simulation</code></em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. -->
<p>
<p align="center">
		<em>Developed by Tomer Kordonsky and Tommy Mordo</em>
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#overview)
- [ Features](#features)
- [ Repository Structure](#repository-structure)
- [ Modules](#modules)
- [ Getting Started](#getting-started)
  - [ Installation](#installation)
  - [ Usage](#usage)
  - [ Input File](#input-file)
- [ Project Roadmap](#project-roadmap)
- [ Contributing](#contributing)
- [ Acknowledgments](#acknowledgments)
</details>
<hr>

##  Overview

<p>The project is a competition-based framework that simulates a ranking game using Large Language Models (LLMs) and custom ranking algorithms. The competition involves multiple "agents," each behaving as a distinct character (e.g., BSc student, professional writer, data science professor), that iteratively edit documents to optimize their ranking for specific search queries. The process involves several rounds where agents generate and rank documents, with feedback provided based on previous rankings to guide future edits.</p>

---

##  Features

1. **Multi-Agent Competition Framework**  
   Simulates a competitive environment where multiple agents, each with a distinct character and behavior profile, work to optimize document rankings. Agents generate and edit documents based on specific prompts and receive feedback to guide their subsequent edits.

2. **Custom Ranking Algorithms**  
   Integrates custom ranking models, such as the E5 model, to evaluate and rank documents based on their relevance to specific queries. The ranking process is crucial for determining the success of each agent's strategies and providing actionable feedback.

3. **Flexible Execution Options**  
   Offers the flexibility to execute the competition either round-by-round or game-by-game:
   - **Round-by-Round**: Games history is incorporated after each round, allowing agents to adjust their strategies continuously. This option can be extended to an online game with humans.
   - **Game-by-Game**: The entire game is played out over multiple rounds before games history is given, allowing a full assessment of strategies before any adjustments.

4. **Feedback Mechanism**  
   Implements a feedback system that provides agents with detailed insights from previous rounds. We implemented different types of prompt to feedback an LLM-based agent. This feedback helps agents refine their strategies and improve document rankings iteratively.

5. **Large Language Model Integration**  
   Leverages powerful LLMs (Large Language Models) like Llama to generate document tailored to improve rankings. These models can be configured with specific parameters, such as temperature and token limits, to produce optimized results.

6. **TREC Text Parsing and Management**  
   Facilitates the extraction and management of documents and queries from TREC text files. This feature ensures that relevant data is accurately parsed and prepared for use in the competition.

7. **Automated Game History Logging**  
   Automatically logs each stage of the competition, including document generations, rankings, and feedback. This logging is crucial for transparency and allows for detailed post-competition analysis.

8. **Modular and Extensible Design**  
   Features a modular architecture that allows easy customization and extension of core components, including agents, LLMs, and ranking algorithms. This design supports adaptability for various research needs.

9. **Error Handling and Robust Logging**  
   Ensures smooth operation through robust error handling and detailed logging, capturing critical information for debugging and performance monitoring.

10. **Comprehensive Output Generation**  
    Produces detailed outputs post-competition, including TREC-style text files and CSV logs, providing a complete record of the competition for further analysis and review.

---

##  Repository Structure

```sh
└── LEMSS/
    ├── LLMs
    │   ├── LLM.py
    │   ├── __init__.py
    │   ├── hugging_face_llm.py
    │   └── mlx_llm.py
    ├── agents
    │   ├── LLM_agent.py
    │   ├── __init__.py
    │   └── agent.py
    ├── competition
    │   ├── __init__.py
    │   ├── competition.py
    │   ├── game.py
    │   └── prompt_manager.py
    ├── constants
    │   ├── __init__.py
    │   └── constants.py
    ├── data
    │   ├── web_track
    │   │   ├── full-topics.xml
    │   │   ├── trec2013-topics.xml
    │   │   ├── trec2014-topics.xml
    │   │   ├── wt09.topics.full.xml
    │   │   ├── wt2010-topics.xml
    │   │   ├── wt2012.xml
    │   └── initial_documents.trectext
    ├── extra
    │   └── project_logo.png
    ├── outputs
    │   └── <(config_md5_hash)_year-month-day>
    │        ├── logs
    │        │   ├── <(logger_name).log>
    │        ├── config.json
    │        ├── competition_history.csv
    │        └── output.trectext
    ├── parsers
    │   ├── __init__.py
    │   ├── query_parser.py
    │   └── trec_parser.py
    ├── players
    │   ├── __init__.py
    │   ├── llm_player.py
    │   └── player.py
    ├── rankers
    │   ├── __init__.py
    │   ├── contriever.py
    │   ├── e5.py
    │   ├── embedding_ranker.py
    │   ├── index_ranker.py
    │   ├── okapi.py
    │   └── ranker.py
    ├── utils
    │   ├── __init__.py
    │   ├── logger.py
    │   └── utils.py
    ├── config.json
    ├── main.py
    ├── readme.md
    ├── outputs
    └── requirements.txt
```

---

## Modules

<details closed><summary>LLMs</summary>

| File                      | Summary                         |
| ---                       | ---                             |
| [LLM.py](LLMs/LLM.py)     | Defines the abstract base class for large language models (LLMs) used in the competition. |
| [hugging_face_llm.py](LLMs/hugging_face_llm.py) | Implements an LLM using the Hugging Face Transformers library for generating and ranking documents. |
| [mlx_llm.py](LLMs/mlx_llm.py) | Implements an LLM using the MLX library for generating and ranking documents. |

</details>

<details closed><summary>agents</summary>

| File                           | Summary                         |
| ---                            | ---                             |
| [LLM_agent.py](agents/LLM_agent.py) | Implements an agent that utilizes large language models (LLMs) for generating and ranking documents. |
| [agent.py](agents/agent.py)    | Provides an abstract base class for defining agents in the competition, including methods for generating and ranking documents. |

</details>

<details closed><summary>competition</summary>

| File                                               | Summary                         |
| ---                                                | ---                             |
| [game.py](competition/game.py)                     | Orchestrates the execution of individual game rounds, handling document generation, ranking, and feedback. |
| [competition.py](competition/competition.py)       | Manages the overall competition setup, execution, and aggregation of game histories across multiple agents. |
| [prompt_manager.py](competition/prompt_manager.py) | Manages the construction of system and user prompts for guiding the LLMs in document generation. |

</details>

<details closed><summary>constants</summary>

| File                                   | Summary                         |
| ---                                    | ---                             |
| [constants.py](constants/constants.py) | Defines constants and mappings used across the project. |

</details>

<details closed><summary>data</summary>

| File                                               | Summary                         |
| ---                                                | ---                             |
| [web_track](data/web_track)                        | Contains TREC web track data files, including topics and initial documents. |
| [initial_documents](data/initial_documents.trectext) | Contains initial documents in TREC text format. |

</details>

<details closed><summary>extra</summary>

| File                                     | Summary                         |
| ---                                      | ---                             |
| [project_logo.png](extra/project_logo.png) | Contains the project logo image used in the README. |

</details>

<details closed><summary>outputs</summary>

| File                                     | Summary                         |
| ---                                      | ---                             |
| [output.trectext](outputs/output.trectext) | Contains the output of the competition in TREC text format, including document rankings and feedback. |
| [config.json](outputs/config.json)       | Contains the configuration settings for the competition, including the number of rounds, agents, and feedback options. |
| [competition_history.csv](outputs/competition_history.csv) | Logs the history of the competition, including agent actions, rankings, and feedback. |

</details>

<details closed><summary>parsers</summary>

| File                                       | Summary                         |
| ---                                        | ---                             |
| [query_parser.py](parsers/query_parser.py) | Parses queries from XML files and TREC text data, integrating queries with corresponding documents. |
| [trec_parser.py](parsers/trec_parser.py)   | Manages the creation of TREC text files from game history data, enabling further analysis and compatibility with TREC tools. |

</details>

<details closed><summary>players</summary>

| File                             | Summary                         |
| ---                              | ---                             |
| [llm_player.py](players/llm_player.py) | Implements a player that utilizes large language models (LLMs) for generating documents. |
| [player.py](players/player.py)   | Provides an abstract base class for defining players in the competition, including methods for generating and receive ranking documents. |

</details>

<details closed><summary>rankers</summary>

| File                           | Summary                                                                                                                          |
| ---                            |----------------------------------------------------------------------------------------------------------------------------------|
| [e5.py](rankers/e5.py)         | Implements the E5 ranking model for evaluating and scoring documents based on query relevance.                                   |
| [contriever.py](rankers/contriever.py) | Implements the Contriever ranking model for evaluating and scoring documents based on query relevance.                           |
| [embedding_ranker.py](rankers/embedding_ranker.py) | Implements an abstract neural ranking model based on document embeddings for evaluating and scoring documents based on query relevance. |
| [index_ranker.py](rankers/index_ranker.py) | Implements an abstract classical ranking model based on document indexing for evaluating and scoring documents based on query relevance.           |
| [okapi.py](rankers/okapi.py)   | Implements the Okapi BM25 ranking model for evaluating and scoring documents based on query relevance.                           |
| [ranker.py](rankers/ranker.py) | Provides an abstract base class for implementing custom ranking models and includes a tie-breaking mechanism.                    |

</details>

<details closed><summary>utils</summary>

| File                         | Summary                                                                                                     |
| ---                          |-------------------------------------------------------------------------------------------------------------|
| [logger.py](utils/logger.py) | Provides a utility for setting up custom loggers to track execution, errors, and other runtime information. |
| [utils.py](utils/utils.py)   | Contains utility functions for common tasks, such as file I/O. |


</details>

<details closed><summary>.</summary>

| File                                 | Summary                         |
| ---                                  | ---                             |
| [README.md](README.md)               | Provides an overview of the project, including instructions on setup, usage, and features. |
| [requirements.txt](requirements.txt) | Lists the required Python packages and dependencies for the project. |
| [main.py](main.py)                   | Entry point for starting the competition, initializing the configuration, and executing the main logic. |
| [config.json](config.json)           | Contains the configuration settings for the competition, including the number of rounds, agents, and feedback options. | 

</details>


---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.10+`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the LEMSS repository:
>
> ```console
> $ git clone ../LEMSS
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd LEMSS
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```
> 
> 4. Update all dependencies:
> ```console
> $ pip install --upgrade -r requirements.txt
> ```

###  Usage
> 1. Update the `config.json` file with the desired competition settings.
> 
> 2. Run LEMSS using the command below:
> ```console
> $ python main.py --config_file config.json
> ```

### Input File
`config.json` default template
```json
{
    "competition": {
        "round_by_round": true,
        "init_docs_path": {
            "queries_folder_path": "data/web_track",
            "docs_file_path": "data/initial_documents.trectext"
        },
        "rankers": {
            "contriever": {
                "model_name": "facebook/contriever"
            }
        },
        "llm": {
            "model_name": "google/gemma-2-9b-it",
            "token": "<ADD TOKEN>",
            "temperature": 0.5,
            "top_p": 0.9
        }
    },
    "game": {
        "max_tokens": 256,
        "rounds": 31,
        "force_max_tokens": true
    },
    "agents": {
        "student": {
            "character": "behave like a BSc student",
            "prompt_format": "Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest). Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits. Focus on editing the most impactful sentences to enhance ranking potential. Target an edited document length of around 147 words, not exceeding 150 words. Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.\n",
            "pairwise": true,
            "depth": 3
        },
        "writer": {
            "character": "behave like a professional writer",
            "prompt_format": "Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest). Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits. Focus on editing the most impactful sentences to enhance ranking potential. Target an edited document length of around 147 words, not exceeding 150 words. Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.\n",
            "pairwise": true,
            "depth": 3
        },
        "editor": {
            "character": "behave like a professional editor",
            "prompt_format": "Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest). Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits. Focus on editing the most impactful sentences to enhance ranking potential. Target an edited document length of around 147 words, not exceeding 150 words. Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.\n",
            "pairwise": true,
            "depth": 3
        },
        "teacher": {
            "character": "behave like an English teacher",
            "prompt_format": "Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest). Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits. Focus on editing the most impactful sentences to enhance ranking potential. Target an edited document length of around 147 words, not exceeding 150 words. Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.\n",
            "pairwise": true,
            "depth": 3
        },
        "professor": {
            "character": "behave like a Data Science professor",
            "prompt_format": "Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest). Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits. Focus on editing the most impactful sentences to enhance ranking potential. Target an edited document length of around 147 words, not exceeding 150 words. Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.\n",
            "pairwise": true,
            "depth": 3
        }
    }
}
```

### competition:
- `competition`: 
    - `round_by_round`: Boolean value to determine if the competition should be executed round-by-round or game-by-game.
    - `init_docs_path`: Path to the initial documents and queries folder.
    - `rankers`: Ranker settings for the competition (there are currently three types of rankers: `contriever`, `e5`, and `okapi`. other rankers can be easily implemented into our code-base).
        1. `contriever`: Contriever ranker settings:
            - `model_name`: The hugging face link to the Contriever model.
        2. `e5`: E5 ranker settings:
            - `model_name`: The hugging face link to the E5 model.
        3. `okapi`: Okapi ranker settings (it uses wikir/en59k as corpus):
            - `index_name`: The name for the index folder to be created.
        
    - `llm`: Large Language Model (LLM) configuration settings.
        - `model_name`: The hugging face link to the LLM model.
        - `token`: Hugging face token for LLM generation.
        - `temperature`: Temperature parameter for LLM generation.
        - `top_p`: Top-p parameter for LLM generation.
        - You can add any other LLM parameters here that is part of the model Hugging Face model.
      
### game:
- `game`: 
    - `max_tokens`: Maximum number of tokens allowed for the LLM to generate.
    - `rounds`: Number of rounds to be executed
    - `force_max_tokens`: Boolean value to determine if the output of the LLM should be trimmed to the `max_tokens` value.

### agents:
- `agents`: 
    - `<name of the agent>`: Name of the agent.
        - `character`: Description of the agent's character.
        - `prompt_format`: The prompt format for the agent.
        - `pairwise`: Boolean value to determine if pairwise or listwise feedback should be provided.
        - `depth`: Depth of the pairwise/ listwise feedback (how many previous rounds should be considered).
    - You can add unlimited amount of agents with different settings.



---

##  Project Roadmap

- [X] `► Build the project structure and core modules.`
- [X] `► Add more ranking models and LLMs.`
- [ ] `► Add penalty system for agents.`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/tomer92808888/Doc-Wars/issues)**: Submit bugs found or log feature requests for the `LEMSS` project.
- **[Submit Pull Requests](https://github.com/tomer92808888/Doc-Wars/pulls)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/tomer92808888/Doc-Wars/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your local account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone ../LEMSS
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to local**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

---


##  Acknowledgments

- Tommy Mordo

---
