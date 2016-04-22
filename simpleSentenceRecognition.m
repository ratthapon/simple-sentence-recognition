%% simple sentence classification using discrete HMM
% this script demostrate the language model framework using discrete HMM.
% including how to train HMM
% how to coding sentences to word numeric data
% how to evaluate HMMs using Viterbi algorithm
%
% input format
%    sentence_1, intent
%    sentence_2, intent
%    ...
% example
%    Hello world, greeting
%    Hi how do you do , greeting
%    Cloud you run a script for me, run_script
%    please run script, run_script
%    pleas tell me the result of script, show_script
%
%  Dependencies:
% 
% Rattaphon Hokking(rathapon_h@outlook.com) 21/4/2016
% 

dataSet = importdata('F:\Project\Alice\sentence.txt');
sentence = {};
intent = {};
%% split sentence and intent
for i = 1:size(dataSet, 1)
    temp = regexp(dataSet{i},':','split');
    sentence(i) = {temp(1)};
    intent(i) = strrep(temp(2),' ', '');
end

%% create system dictionary
dict = {};
for i = 1:size(sentence, 2)
    temp = regexp(sentence{i},' ','split');
    dict = [dict temp{:}];
end
dict = unique(dict);

%% create intent dictionary
intentDict = unique(intent);

%% HMM model for sequence
% init hmm
hmm = {};
nState = 3;
nObserver = size(dict, 2);
nModel = size(intentDict, 2);
for i = 1:nModel
    % random guess model, prevent zero transition
    transition = rand(nState, nState);
    emission = rand(nState, nObserver);
    
    % norm range to 1. each state must trans to another states
    transition = transition./ repmat(sum(transition, 2), 1, nState);
    emission = emission./ repmat(sum(emission, 2), 1, nObserver);
    
    % store the model
    hmm{i} = [{transition},{emission}];
end

%% define sentence coding
% code word to numeric
code = @(w,dictionary)  find(not(cellfun('isempty', strfind(dictionary, w))));

%% hmm training
for i = 1:size(sentence, 2)
    % encode sentence to idx sequence
    words = regexp(sentence{i},' ','split');
    words = words{1};
    seq = zeros(size(words, 2),1);
    
    % map words to numeric sequence
    for j = 1:size(words, 2)
        wordIdx = code(words{j}, dict);
        seq(j) = wordIdx;
    end
    % find the expect intent
    intentIdx = code(intent{j}, intentDict);
    
    % load previous model
    ESTTR = hmm{intentIdx}{1};
    ESTEMIT = hmm{intentIdx}{2};
    % peudotransiiton parameters to prevent zero divide, zero transition
    pseudoEmit = ESTEMIT;
    pseudoTr = ESTTR;
    
    %% find new transition and emission
    [ESTTR,ESTEMIT] = hmmtrain( seq', ...
        ESTTR,ESTEMIT,...
        'algorithm','viterbi',...
        'PSEUDOTRANSITIONS',pseudoTr, ...
        'PSEUDOEMISSIONS',pseudoEmit);
    
    hmm{intentIdx} = [{ESTTR},{ESTEMIT}];
end

%% evaluate HMM
labelVec = zeros(size(sentence, 2), 1);
LOGPSEQBuffer = zeros(size(hmm,2), size(sentence, 2));
for i = 1:size(sentence, 2)
    % encode sentence to idx sequence
    words = regexp(sentence{i},' ','split');
    words = words{1};
    seq = zeros(size(words, 2),1);
    for j = 1:size(words, 2)
        wordIdx = code(words{j}, dict);
        seq(j) = wordIdx;
    end
    intentIdx = code(intent{j}, intentDict);
    labelVec(i) = intentIdx;
    
    % compare log likelihood of the sequence with models
    for mIdx = 1:size(hmm,2)
        model = hmm{mIdx}; % point to model
        ESTTR = model{1};
        ESTEMIT = model{2};
        % find probability of single sequence
        [~, logProbSeq] = hmmviterbi(seq',ESTTR,ESTEMIT);
        LOGPSEQBuffer(mIdx,i) = logProbSeq;
    end
end
%% find the maximum log likelihood for each sequence
[~,outClass] = max(LOGPSEQBuffer);

%% calculate accuracy rate
accRate = sum(outClass == labelVec')/size(labelVec',2);

