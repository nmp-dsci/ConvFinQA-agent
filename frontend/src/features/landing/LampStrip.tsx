import { Lamp } from '@/components/console';
import type { LampTone } from '@/components/console';
import type { BoardData } from './useBoardData';
import { versionLabel } from '../admin/lib';

/**
 * Mode, champion, gate — the three facts that decide how to read every other
 * number on the board.
 *
 * Mode is first and is the only lamp whose *shape* changes: a solid green ring
 * means this deployment holds a key and answers with the model; a dashed amber
 * ring means it holds no key at all and chat is replayed from recordings.
 * Everything below the strip has to be read in that light, which is why it
 * sits above the tiles rather than beside them.
 */
export function LampStrip({ board }: { board: BoardData }) {
  const { health, isDemo, servingChampion, servingRuntime, campaigns } = board;

  /**
   * The gate lamp reports the campaign's most recent verdict.
   *
   * It used to run the legacy comparator over the 770-question corpus and
   * report *that* rule's answer, which since the campaign protocol is the wrong
   * question asked of the wrong population: it applied a net-positive rule the
   * loop retired, to a corpus the loop does not gate on, about versions the
   * loop rolled back. A lamp that reads "pass" under a rule nothing promotes on
   * is worse than no lamp.
   */
  const gateLamp = (() => {
    const experiments = campaigns?.experiments ?? [];
    const latest = experiments[experiments.length - 1];
    if (!latest) {
      return {
        value: 'no challenger',
        tone: 'idle' as LampTone,
        dashed: true,
        tooltip:
          'No experiment has been gated yet. The gate promotes a challenger only when it is net positive on the shared gate questions AND clears one-sided cluster-corrected McNemar at α = 0.05.',
      };
    }
    const p = latest.cluster_p_one_sided;
    const pText = p == null ? '—' : p.toFixed(3);
    const delta =
      latest.accuracy_delta == null
        ? '—'
        : `${latest.accuracy_delta >= 0 ? '+' : ''}${(latest.accuracy_delta * 100).toFixed(2)}pp`;
    if (latest.promoted) {
      return {
        value: `${latest.label} promoted`,
        tone: 'good' as LampTone,
        dashed: false,
        tooltip: `${latest.baseline_version} → ${latest.candidate_version} by rewriting ${latest.target_agent} alone: ${delta} on the gate split, one-sided clustered McNemar p = ${pText}.`,
      };
    }
    return {
      value: `${latest.label} refused`,
      tone: 'bad' as LampTone,
      dashed: false,
      tooltip: (
        <>
          <div className="mb-1 font-medium">The gate is working, not broken.</div>
          {latest.candidate_version} rewrote {latest.target_agent} and moved the gate split by{' '}
          {delta} — {latest.fixed ?? 0} questions fixed against {latest.broken ?? 0} broken — but at
          p = {pText} that is not distinguishable from noise at α = 0.05. Promotion needs net
          positive <em>and</em> significance; net positive alone promoted three versions whose
          confidence intervals contained zero.
        </>
      ),
    };
  })();

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Lamp
        label="mode"
        value={isDemo ? 'replay · keyless' : 'live'}
        tone={isDemo ? 'amber' : 'good'}
        dashed={isDemo}
        to="/admin/system"
        tooltip={
          isDemo
            ? 'This deployment holds no API key. Chat replays conversations recorded in development through the same events a live turn emits. Nothing on this page was measured against live production traffic.'
            : `Live against ${health?.bundle.lm_mini ?? 'the champion model'}. Turns are answered by the model, not replayed.`
        }
      />
      <Lamp
        label="champion"
        value={versionLabel(servingChampion ?? campaigns?.sdk_champion ?? campaigns?.champion) || 'unset'}
        tone="info"
        to="/admin/runtimes"
        tooltip={
          health
            ? `${servingRuntime === 'agent_sdk' ? 'One Claude Agent SDK session per conversation, the six calculator tools as its only tools' : 'Four prompted agents in a fixed order'} · sdk_champion ${health.sdk_champion ?? 'unset'} · pipeline champion ${health.champion ?? 'unset'} · dataset ${health.bundle.dataset_hash} · code ${health.bundle.code_sha}.`
            : 'The version currently serving.'
        }
      />
      <Lamp
        label="gate"
        value={gateLamp.value}
        tone={gateLamp.tone}
        dashed={gateLamp.dashed}
        to="/admin/experiments"
        tooltip={gateLamp.tooltip}
      />
    </div>
  );
}
