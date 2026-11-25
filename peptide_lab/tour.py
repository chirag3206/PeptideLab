"""In-dashboard walkthrough. Each stop opens the corresponding workspace."""
import streamlit as st


STOPS = {
    'Overview': ('Welcome to PeptideLab', 'Follow this tour to try the dashboard with the included data. In the sidebar, **Prediction engine** selects the models and **Ranking rule** controls how activity and toxicity contribute to rank.'),
    'Dataset explorer': ('Try a measured peptide', 'Choose **Activity benchmark** and the **test** split. Keep the preselected peptides and click **Predict dataset activity & toxicity**. Compare the predictions with the recorded labels.'),
    'Target-specific screening': ('Compare general and organism-specific scores', 'Choose an organism and **Dataset peptides**, then click **Screen for selected target**. Compare the two ranks in one row, or hover over a point on the plot. **Overall results** shows which organisms have evaluated models.'),
    'Generate & rank': ('Explore a generated batch', 'Click **Generate, screen & rank** to sample the VAE. See how many peptides passed the filters, inspect their ranks, and download the full table.'),
    'Screen sequences': ('Use your own input', 'Try the example sequences, paste FASTA text, or upload a file. Click **Screen sequences** to see activity, toxicity and rank. Rerun after changing the engine or ranking rule.'),
    'Prediction history': ('Revisit a run', 'When you run a prediction, it appears here with its original settings. Choose a run to inspect or download it. History lasts for this browser session.'),
    'Model results': ('Check the evidence', 'Choose **Compact baseline**, **ESM-2 + CNN**, or **Engine comparison** to inspect held-out results. Compact predictors and the **VAE generator** also have hyperparameter tuning tables.'),
}


def go_to_step(index):
    st.session_state.tour_step = index
    st.session_state.workspace = list(STOPS)[index]


def show_tour(page):
    index = st.session_state.setdefault('tour_step', 0)
    if index is not None and list(STOPS)[index] != page:
        st.session_state.tour_step = index = None
    if index is None:
        st.sidebar.button('Replay dashboard tour', on_click=go_to_step, args=(0,))
        return
    title, instruction = STOPS[page]
    with st.container(border=True):
        st.caption(f'Dashboard tour · {index + 1} of {len(STOPS)}')
        st.subheader(title)
        st.markdown(instruction)
        back, forward, dismiss = st.columns(3)
        back.button('Back', key='tour_back', disabled=index == 0, on_click=go_to_step, args=(index - 1,))
        if index + 1 < len(STOPS):
            forward.button('Next', key='tour_next', on_click=go_to_step, args=(index + 1,), type='primary')
        else:
            forward.button('Finish tour', key='tour_finish', on_click=st.session_state.update,
                           args=({'tour_step': None},), type='primary')
        dismiss.button('Skip tour', key='tour_skip', on_click=st.session_state.update, args=({'tour_step': None},))
