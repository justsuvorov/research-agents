# Spec — OpuLoadCalculator

## Responsibility

Deterministic calculation of loads on the slewing bearing (ОПУ) of a ship crane and of the
torque that the slewing drive must overcome. Replaces the former LLM-based
`EngineeringCalculator`: all formulas are implemented in code (dissertation, section 2.1.4),
no external services are called, the same input always gives the same output.

The resulting rows form the initial table for the regression model of the torque
(`M` as target, `WEIGHT`, `PSI`, `Q` as features).

---

## Config Schema — `data.opu_load_calculations`

```yaml
data:
  opu_load_calculations:
    - name: ship_cranes_opu
      slew_angles_deg: [0, 45, 90]        # boom angle from the ship centreline
      load_cases:
        - {name: port_lifting, psi: 1.25, wind_pressure_pa: 125, heel_deg: 5}
        - {name: sea_slewing,  psi: 1.66, wind_pressure_pa: 400, heel_deg: 5}
      cranes:
        - name: km_10m
          crane_type: M
          boom_mass_t: 1.196
          reach_m: 10
          cargo_t: [2.2, 2.75]
          slew_speed_rpm: 1.33
          slew_start_time_s: 3
          bearing_track_diameter_m: 1.0
          bearing_friction: 0.004
          gear: {pinion_teeth: 12, ring_teeth: 95, module_mm: 18}
```

Grid: cranes × cargo_t × load_cases × slew_angles_deg — one row per combination.

---

## Formulas

Units: masses in t, lengths in m, forces in kN, moments in kN·m, g = 9.81 m/s².
θ — heel (base inclination), φ — boom angle from the centreline.

Gravity forces: G_Q = ψ·m_Q·g, G_P = m_P·g, G_R = m_R·g, ΣG = G_Q + G_P + G_R.
Static moment of masses about the slewing axis: S = G_Q·R + G_P·x_P + G_R·x_R.
Wind force: F_w = q·(A_Q + A_P)/1000.

| Value | Formula | Source |
|---|---|---|
| Axial force F_a | ΣG·cos θ | 2.1.4 (sum of gravity forces) |
| Radial force F_r | ΣG·sin θ + F_w | 2.1.4 (heel and wind) |
| Overturning moment M_опр | S·cos θ + ΣG·h_G·sin θ + F_w·h_w | 2.1.4 |
| Friction torque in ОПУ M_тр | μ/2·(4.4·M_опр + F_a·D_L + 3.81·F_r·D_L) | (2.36) |
| Heel torque M_θ | S·sin θ·\|cos φ\| | 2.1.4 (heel component) |
| Wind torque M_w | q/1000·(A_Q·R + A_P·x_P)·\|cos φ\| | 2.1.4 (wind on board) |
| Inertia torque M_ин | J·ε/1000, J = 1000·(m_Q·R² + m_P·(x_P² + R²/12) + m_R·x_R²), ε = 2πn/(60·t_п) | 2.1.4 (start-up inertia) |
| Slewing torque M_кр | M_тр + M_θ + M_w + M_ин | 2.1.4 |
| Pinion shaft torque T₂ | M_кр·γ_m/(i_зп·η_зп·n_пр·η_н), i_зп = z₂/z₁ | (2.37) |
| Normal tooth force F_n | 2·T₂/(m·z₁·cos α_w) | (2.38) |

---

## Output Row

| Column | Meaning |
|---|---|
| crane, TYPE | crane name and type (M, T, P) |
| load_case | load case name |
| WEIGHT | cargo mass, t |
| PSI | dynamic coefficient ψ |
| Q | boom mass, t |
| heel_deg, slew_deg, wind_pa, reach_m | case geometry and wind |
| F_a_kN, F_r_kN, M_opr_kNm | loads on ОПУ |
| M_tr_kNm, M_heel_kNm, M_wind_kNm, M_in_kNm | torque components |
| M | slewing torque M_кр, kN·m (regression target) |
| T2_kNm, F_n_kN | pinion torque and normal tooth force (input for wear calculation) |
| source, source_type | rule name, `opu_load_calculation` |

---

## Success Criteria

- No network calls, no LLM.
- Number of rows = Σ over cranes of len(cargo_t) × len(load_cases) × len(slew_angles_deg).
- With θ = 0, q = 0 and ε → 0 the torque equals the friction torque (2.36).
- Torque components from heel and wind decrease as |cos φ|.
