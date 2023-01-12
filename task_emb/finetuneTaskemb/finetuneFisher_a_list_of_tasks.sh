#!bin/bash

TASKS='acronym_identification ade_corpus_v2-classification ade_corpus_v2-dosage ade_corpus_v2-effect adversarialqa aeslc ag_news ai2_arc amazon_polarity anli app_reviews aqua_rat art aslg_pc12 biomrc blimp-anaphor_gender_agreement blimp-anaphor_number_agreement blimp-determiner_noun_agreement_with_adj_irregular_1 blimp-ellipsis_n_bar_1 blimp-ellipsis_n_bar_2 blimp-existential_there_quantifiers_1 blimp-irregular_past_participle_adjectives blimp-sentential_negation_npi_licensor_present blimp-sentential_negation_npi_scope blimp-wh_questions_object_gap boolq break-QDMR break-QDMR-high-level circa climate_fever codah common_gen commonsense_qa cos_e cosmos_qa crawl_domain crows_pairs dbpedia_14 definite_pronoun_resolution discovery dream duorc e2e_nlg_cleaned eli5-askh eli5-asks eli5-eli5 emo emotion empathetic_dialogues ethos-directed_vs_generalized ethos-disability ethos-gender ethos-national_origin ethos-race ethos-religion ethos-sexual_orientation financial_phrasebank freebase_qa gigaword glue-cola glue-mnli glue-mrpc glue-qnli glue-qqp glue-rte glue-sst2 glue-wnli google_wellformed_query hate_speech18 hate_speech_offensive hatexplain health_fact hellaswag hotpot_qa imdb jeopardy kilt_ay2 kilt_fever kilt_hotpotqa kilt_nq kilt_trex kilt_wow kilt_zsre lama-conceptnet lama-google_re lama-squad lama-trex liar limit math_qa mc_taco medical_questions_pairs mocha multi_news numer_sense onestop_english openbookqa paws piqa poem_sentiment proto_qa qa_srl qasc quail quarel quartz-no_knowledge quartz-with_knowledge quoref race-high race-middle reddit_tifu-title reddit_tifu-tldr ropes rotten_tomatoes samsum scicite sciq scitail search_qa sick sms_spam social_i_qa spider squad-no_context squad-with_context superglue-cb superglue-copa superglue-multirc superglue-record superglue-rte superglue-wic superglue-wsc swag tab_fact trec trec-finegrained tweet_eval-emoji tweet_eval-emotion tweet_eval-hate tweet_eval-irony tweet_eval-offensive tweet_eval-sentiment tweet_eval-stance_abortion tweet_eval-stance_atheism tweet_eval-stance_climate tweet_eval-stance_feminist tweet_eval-stance_hillary tweet_qa web_questions wiki_auto wiki_bio wiki_qa wiki_split wikisql wino_grande wiqa xsum yahoo_answers_topics yelp_polarity yelp_review_full'
TASKS='imdb'
#2500
DATADIR='/home/juanzha/projects/crossfit_data_v2'
OURDIR='taskemb'



#calculate fisher information
for TASK in $TASKS
do

echo "-------------------------Task: $TASK -----------------"

python tune_fixed_singletask_adapter.py \
--task_dir ${DATADIR}/${TASK}/ \
--do_train \
--learning_rate_list 3e-5 \
--bsz_list 8 \
--total_steps 2500 \
--eval_period 100 \
--warmup_steps 150 \
--max_grad_norm 0.1 \
--weight_decay 0.01 \
--model facebook/bart-base \
--output_dir ${OURDIR}/singletask-${TASK} \
--gradient_accumulation_steps 1 \
--predict_batch_size 1
done



