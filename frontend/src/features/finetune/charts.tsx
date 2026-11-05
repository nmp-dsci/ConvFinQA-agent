/**
 * The fine-tuning figures (s32): hand-drawn SVG on the kit's `.dia` vocabulary, every colour a
 * token (`finetune.css`). Each is never drawn narrower than its viewBox — on a phone it scrolls in
 * its own wrapper rather than shrinking 13px labels under the house's 12.8px floor — and each crops
 * its y axis to the values it holds and says so in its caption, so a dot's height carries the value
 * rather than a bar's length.
 */
import type { FineTuningRecord, FtPass, FtRound } from './api';
import { checkpointRows, epochLabel, passIndex, pct, pp, stageLines } from './model';

function domain(values: number[], pad = 0.02): [number, number] {
  const lo = Math.floor((Math.min(...values) - pad) * 20) / 20;
  const hi = Math.ceil((Math.max(...values) + pad) * 20) / 20;
  return [lo, hi];
}

function ticks(lo: number, hi: number, step = 0.05): number[] {
  const out: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
  return out;
}

const num = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v);

/**
 * Every stage of the story on test 100 (solid) and val (hollow), base to the last round, with the
 * teacher's test score as a reference line. A harness change is a stage of its own, so each segment
 * of the line is one change.
 */
export function ProgressChart({ record }: { record: FineTuningRecord }) {
  const passes = passIndex(record);
  const test = record.progression.test;
  const val = record.progression.val;
  const teacher = record.teacher ? passes.get(record.teacher.pass) : undefined;
  const accOf = (id: string | null) => (id ? passes.get(id)?.accuracy ?? null : null);
  const values = [...test, ...val].map((s) => accOf(s.pass)).filter(num);
  if (teacher?.accuracy != null) values.push(teacher.accuracy);
  if (!values.length) return <p className="empty">— no stage has a complete pass to draw</p>;

  const n = test.length;
  const [W, H, left, right, top, bottom] = [960, 340, 56, 190, 24, 64];
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const [lo, hi] = domain(values);
  const slot = plotW / n;
  const x = (i: number) => left + slot * (i + 0.5);
  const y = (v: number) => top + ((hi - v) / (hi - lo)) * plotH;
  const line = (stages: typeof test) =>
    stages
      .map((s, i) => [i, accOf(s.pass)] as const)
      .filter(([, v]) => num(v))
      .map(([i, v]) => `${x(i).toFixed(1)},${y(v as number).toFixed(1)}`)
      .join(' ');
  const last = test.reduce((k, s, i) => (accOf(s.pass) != null ? i : k), -1);

  return (
    <div className="figscroll">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="dia ft-svg"
        style={{ minWidth: W }}
        role="img"
        aria-label={`Accuracy at each of ${n} stages on test 100 and val, with the teacher's test score`}
        data-testid="ft-progress-chart"
      >
        <title>Student accuracy at each stage of fine-tuning</title>
        {ticks(lo, hi).map((v) => (
          <g key={v}>
            <line className="gl" x1={left} x2={W - right} y1={y(v)} y2={y(v)} />
            <text className="tx k" x={left - 8} y={y(v) + 4} textAnchor="end">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {teacher?.accuracy != null && (
          <g className="ft-ref">
            <line x1={left} x2={W - right} y1={y(teacher.accuracy)} y2={y(teacher.accuracy)} />
            <text className="tx" x={W - right + 10} y={y(teacher.accuracy) + 4}>
              teacher {pct(teacher.accuracy)}
            </text>
            <text className="tx k" x={W - right + 10} y={y(teacher.accuracy) + 20}>
              {record.teacher?.name}, test
            </text>
          </g>
        )}
        <polyline className="ft-line val" points={line(val)} />
        <polyline className="ft-line test" points={line(test)} />
        {val.map((s, i) => {
          const v = accOf(s.pass);
          return num(v) ? (
            <g key={`v-${s.key}`}>
              <circle className="ft-pt val" cx={x(i)} cy={y(v)} r={4.5} />
              <text className="tx s" x={x(i)} y={y(v) + 20} textAnchor="middle">
                {pct(v)}
              </text>
            </g>
          ) : null;
        })}
        {test.map((s, i) => {
          const v = accOf(s.pass);
          const [a, b] = stageLines(s, record);
          return (
            <g key={`t-${s.key}`}>
              {num(v) ? (
                <>
                  <circle className={`ft-pt test${i === last ? ' hi' : ''}`} cx={x(i)} cy={y(v)} r={5.5} />
                  <text className={`tx${i === last ? ' hi' : ''}`} x={x(i)} y={y(v) - 12} textAnchor="middle">
                    {pct(v)}
                  </text>
                </>
              ) : (
                <text className="tx s" x={x(i)} y={top + plotH / 2} textAnchor="middle">
                  — not on test
                </text>
              )}
              <text className="tx" x={x(i)} y={H - bottom + 26} textAnchor="middle">
                {a}
              </text>
              <text className="tx k" x={x(i)} y={H - bottom + 44} textAnchor="middle">
                {b}
              </text>
            </g>
          );
        })}
        <line className="ax" x1={left} x2={W - right} y1={top + plotH} y2={top + plotH} />
      </svg>
    </div>
  );
}

/**
 * A round's sweep on val: its start, then each checkpoint, with the start's accuracy carried across
 * as a reference so every checkpoint reads as a step from it. The pick is filled.
 */
export function CheckpointChart({ round, record }: { round: FtRound; record: FineTuningRecord }) {
  const passes = passIndex(record);
  const rows = checkpointRows(round, passes);
  const start: FtPass | undefined = round.start.val_pass ? passes.get(round.start.val_pass) : undefined;
  const values = [start?.accuracy, ...rows.map((r) => r.val?.accuracy)].filter(num);
  if (!values.length) return <p className="empty">— no checkpoint of this round was scored on val</p>;

  const cols = [{ key: 'start' }, ...rows];
  const [W, H, left, right, top, bottom] = [900, 300, 56, 28, 28, 64];
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const [lo, hi] = domain(values, 0.03);
  const slot = plotW / cols.length;
  const x = (i: number) => left + slot * (i + 0.5);
  const y = (v: number) => top + ((hi - v) / (hi - lo)) * plotH;

  return (
    <div className="figscroll">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="dia ft-svg"
        style={{ minWidth: W }}
        role="img"
        aria-label={`${round.label}: val accuracy of its start and ${rows.length} checkpoints`}
        data-testid="ft-checkpoint-chart"
      >
        <title>{round.label}: each checkpoint on val against where the round started</title>
        {ticks(lo, hi).map((v) => (
          <g key={v}>
            <line className="gl" x1={left} x2={W - right} y1={y(v)} y2={y(v)} />
            <text className="tx k" x={left - 8} y={y(v) + 4} textAnchor="end">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {num(start?.accuracy) && (
          <line className="ft-startline" x1={x(0)} x2={W - right} y1={y(start!.accuracy!)} y2={y(start!.accuracy!)} />
        )}
        {num(start?.accuracy) && (
          <g>
            <rect className="ft-start" x={x(0) - 5.5} y={y(start!.accuracy!) - 5.5} width={11} height={11} />
            <text className="tx s" x={x(0)} y={y(start!.accuracy!) - 12} textAnchor="middle">
              {pct(start!.accuracy)}
            </text>
          </g>
        )}
        <text className="tx" x={x(0)} y={H - bottom + 26} textAnchor="middle">
          start
        </text>
        <text className="tx k" x={x(0)} y={H - bottom + 44} textAnchor="middle">
          {round.start.label}
        </text>
        {rows.map((r, k) => {
          const i = k + 1;
          const v = r.val?.accuracy;
          return (
            <g key={r.key} data-checkpoint={r.key} data-best={r.best ? 'true' : undefined}>
              {num(v) ? (
                <>
                  <circle className={`ft-pt ck${r.best ? ' hi' : ''}`} cx={x(i)} cy={y(v)} r={r.best ? 6.5 : 5.5} />
                  <text className={`tx${r.best ? ' hi' : ''}`} x={x(i)} y={y(v) - 14} textAnchor="middle">
                    {pct(v)}
                  </text>
                  {r.vsStart && (
                    <text className="tx k" x={x(i)} y={y(v) + 22} textAnchor="middle">
                      {pp(r.vsStart.delta_pp)}
                    </text>
                  )}
                </>
              ) : (
                <text className="tx s" x={x(i)} y={top + plotH / 2} textAnchor="middle">
                  — not scored
                </text>
              )}
              <text className="tx" x={x(i)} y={H - bottom + 26} textAnchor="middle">
                {epochLabel(r.key)}
              </text>
              <text className={`tx ${r.best ? 'hi' : 'k'}`} x={x(i)} y={H - bottom + 44} textAnchor="middle">
                {r.best ? 'picked' : `step ${r.step}`}
              </text>
            </g>
          );
        })}
        <line className="ax" x1={left} x2={W - right} y1={top + plotH} y2={top + plotH} />
      </svg>
    </div>
  );
}

/** Training loss at every logged step, with each saved checkpoint marked where it was written. */
export function LossChart({ round }: { round: FtRound }) {
  const points = round.loss;
  if (!points.length) return <p className="empty">— this run logged no loss</p>;
  const [W, H, left, right, top, bottom] = [900, 260, 56, 28, 34, 44];
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const losses = points.map((p) => p.loss);
  const lo = Math.floor((Math.min(...losses) - 0.02) * 10) / 10;
  const hi = Math.ceil((Math.max(...losses) + 0.02) * 10) / 10;
  const maxStep = Math.max(...points.map((p) => p.step));
  const x = (s: number) => left + ((s - 1) / Math.max(1, maxStep - 1)) * plotW;
  const y = (v: number) => top + ((hi - v) / (hi - lo)) * plotH;
  const step = hi - lo > 0.6 ? 0.2 : 0.1;

  return (
    <div className="figscroll">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="dia ft-svg"
        style={{ minWidth: W }}
        role="img"
        aria-label={`${round.label}: training loss over ${maxStep} steps`}
        data-testid="ft-loss-chart"
      >
        <title>{round.label}: training loss by step</title>
        {ticks(lo, hi, step).map((v) => (
          <g key={v}>
            <line className="gl" x1={left} x2={W - right} y1={y(v)} y2={y(v)} />
            <text className="tx k" x={left - 8} y={y(v) + 4} textAnchor="end">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        {round.checkpoints.map((c) => {
          const best = c.key === round.best;
          return (
            <g key={c.key}>
              <line className={`ft-ckline${best ? ' hi' : ''}`} x1={x(c.step)} x2={x(c.step)} y1={top - 6} y2={top + plotH} />
              <text className={`tx ${best ? 'hi' : 'k'}`} x={x(c.step)} y={top - 12} textAnchor="middle">
                {c.key}
                {best ? ' · picked' : ''}
              </text>
            </g>
          );
        })}
        <polyline className="ft-line loss" points={points.map((p) => `${x(p.step).toFixed(1)},${y(p.loss).toFixed(1)}`).join(' ')} />
        <line className="ax" x1={left} x2={W - right} y1={top + plotH} y2={top + plotH} />
        <text className="tx k" x={left} y={H - 12}>
          step 1
        </text>
        <text className="tx k" x={W - right} y={H - 12} textAnchor="end">
          step {maxStep}
        </text>
      </svg>
    </div>
  );
}
