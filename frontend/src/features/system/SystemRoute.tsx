import { Placeholder } from '../admin/Placeholder';

export default function SystemRoute() {
  return (
    <Placeholder title="System" phase={5} testId="system-route">
      How this deployment is put together and what it refuses to do: the bundle fingerprint, the
      three-layer demo gate (routes, a real disabled fieldset, a server-side 501), the rate and
      in-flight limits, and the tracking/registry wiring. `/debrief` redirects here.
    </Placeholder>
  );
}
