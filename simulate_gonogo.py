% %

pFA = 0.1:0.2: 0.9;

pGo = 0.05:0.15: 0.95; % probability
of
a
Go
trial

nTrials = 150; % number
of
trials
per
simulation

nSims = 10000; % number
of
simulations

clf
t = tiledlayout(length(pFA), length(pGo));
t.TileIndexing = 'columnmajor';
t.TileSpacing = 'none';

clear
ax
for igo = 1:length(pGo)
for ifa = 1:length(pFA)

pNoGo = 1 - pGo(igo);

nGo = round(pGo(igo) * nTrials);
nNoGo = round(pNoGo * nTrials);

hits = randi(nGo, nSims, 1);
miss = pGo(igo) * nTrials - hits;

falseAlarms = randi(round(pFA(ifa) * nTrials), nSims, 1);
correctRejects = pNoGo * nTrials - falseAlarms;

hitRate = hits / nGo;
falseAlarmRate = falseAlarms / nNoGo;

percCorrect = (hits + correctRejects). / nTrials;

% constrain
for hit and false alarm rates to avoid inf values
hitRate = min(max(hitRate, 0.001), 0.999);
falseAlarmRate = min(max(falseAlarmRate, 0.001), .999);

% compute
d
'
dprime = norminv(hitRate) - norminv(falseAlarmRate);

ax(ifa, igo) = nexttile;
plot(percCorrect, dprime, '.k', 'markersize', 1);
xlim([0 1]);
grid
on

if igo == 1
    ylabel({sprintf('p(FA) = %.2f', pFA(ifa)), 'd'''});
end

if ifa == 1
    title(sprintf('p(Go) = %.2f', pGo(igo)))
end

if ifa == length(pFA)
    xlabel('% Correct');
end

end
end

y = cell2mat(get(ax, 'ylim'));
set(ax, 'ylim', [-1 1]. * max(abs(y(:))));
set(ax(1: end - 1,:), 'xticklabel', '');
set(ax(:, 2: end), 'yticklabel', '');





