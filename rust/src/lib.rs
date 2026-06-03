pub type Matrix = Vec<Vec<f64>>;

fn validate_matrix(matrix: &Matrix) -> Result<(usize, usize), String> {
    if matrix.is_empty() {
        return Err("matrix must not be empty".to_string());
    }
    let cols = matrix[0].len();
    if cols == 0 {
        return Err("matrix rows must not be empty".to_string());
    }
    if matrix.iter().any(|row| row.len() != cols) {
        return Err("matrix rows must have equal length".to_string());
    }
    Ok((matrix.len(), cols))
}

fn validate_square(matrix: &Matrix) -> Result<usize, String> {
    let (rows, cols) = validate_matrix(matrix)?;
    if rows != cols {
        return Err("matrix must be square".to_string());
    }
    Ok(rows)
}

fn complete_rows(data: &Matrix) -> Result<Matrix, String> {
    let (_, cols) = validate_matrix(data)?;
    let rows: Matrix = data
        .iter()
        .filter(|row| row.iter().all(|value| value.is_finite()))
        .cloned()
        .collect();
    if rows.len() < 2 {
        return Err("at least two complete observations are required".to_string());
    }
    if rows.iter().any(|row| row.len() != cols) {
        return Err("matrix rows must have equal length".to_string());
    }
    Ok(rows)
}

fn centered(data: &Matrix) -> Result<Matrix, String> {
    let rows = complete_rows(data)?;
    let (observations, cols) = validate_matrix(&rows)?;
    let mut means = vec![0.0; cols];
    for row in &rows {
        for (col, value) in row.iter().enumerate() {
            means[col] += *value;
        }
    }
    for mean in &mut means {
        *mean /= observations as f64;
    }
    Ok(rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .enumerate()
                .map(|(col, value)| value - means[col])
                .collect()
        })
        .collect())
}

fn covariance(centered: &Matrix, denominator: f64, frequency: f64) -> Matrix {
    let cols = centered[0].len();
    let mut out = vec![vec![0.0; cols]; cols];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, cell) in out_row.iter_mut().enumerate().skip(i) {
            *cell =
                centered.iter().map(|row| row[i] * row[j]).sum::<f64>() / denominator * frequency;
        }
    }
    for i in 0..cols {
        let (current_and_before, after) = out.split_at_mut(i + 1);
        let current_row = &current_and_before[i];
        for (offset, row) in after.iter_mut().enumerate() {
            row[i] = current_row[i + 1 + offset];
        }
    }
    out
}

pub fn sample_covariance_matrix(data: Matrix, frequency: f64) -> Result<Matrix, String> {
    let centered = centered(&data)?;
    Ok(covariance(
        &centered,
        (centered.len() - 1) as f64,
        frequency,
    ))
}

pub fn ledoit_wolf_covariance_matrix(
    data: Matrix,
    frequency: f64,
) -> Result<(Matrix, f64), String> {
    let centered = centered(&data)?;
    let observations = centered.len();
    let cols = centered[0].len();
    let empirical = covariance(&centered, observations as f64, 1.0);
    let target_variance = (0..cols).map(|idx| empirical[idx][idx]).sum::<f64>() / cols as f64;
    let mut delta = 0.0;
    for (i, row) in empirical.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            let target = if i == j { target_variance } else { 0.0 };
            delta += (value - target).powi(2);
        }
    }
    let shrinkage = if delta <= 0.0 {
        1.0
    } else {
        let mut beta = 0.0;
        for row in &centered {
            for i in 0..cols {
                for j in 0..cols {
                    beta += (row[i] * row[j] - empirical[i][j]).powi(2);
                }
            }
        }
        (beta / (observations * observations) as f64).min(delta) / delta
    };
    let mut matrix = vec![vec![0.0; cols]; cols];
    for i in 0..cols {
        for j in 0..cols {
            let target = if i == j { target_variance } else { 0.0 };
            matrix[i][j] = ((1.0 - shrinkage) * empirical[i][j] + shrinkage * target) * frequency;
        }
    }
    Ok((matrix, shrinkage.clamp(0.0, 1.0)))
}

pub fn oas_covariance_matrix(data: Matrix, frequency: f64) -> Result<(Matrix, f64), String> {
    let centered = centered(&data)?;
    let observations = centered.len() as f64;
    let cols = centered[0].len();
    let empirical = covariance(&centered, observations, 1.0);
    let mu = (0..cols).map(|idx| empirical[idx][idx]).sum::<f64>() / cols as f64;
    let alpha = empirical
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>()
        / (cols * cols) as f64;
    let denominator = (observations + 1.0) * (alpha - (mu * mu / cols as f64));
    let shrinkage = if denominator <= 0.0 {
        1.0
    } else {
        ((alpha + mu * mu) / denominator).min(1.0)
    };
    let mut matrix = vec![vec![0.0; cols]; cols];
    for i in 0..cols {
        for j in 0..cols {
            let target = if i == j { mu } else { 0.0 };
            matrix[i][j] = ((1.0 - shrinkage) * empirical[i][j] + shrinkage * target) * frequency;
        }
    }
    Ok((matrix, shrinkage.clamp(0.0, 1.0)))
}

pub fn covariance_to_correlation(covariance: Matrix) -> Result<Matrix, String> {
    let size = validate_square(&covariance)?;
    let std: Vec<f64> = (0..size)
        .map(|idx| covariance[idx][idx].max(0.0).sqrt())
        .collect();
    let mut corr = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            let denom = std[i] * std[j];
            corr[i][j] = if denom > 0.0 {
                covariance[i][j] / denom
            } else {
                0.0
            }
            .clamp(-1.0, 1.0);
        }
        corr[i][i] = 1.0;
    }
    Ok(corr)
}

pub fn correlation_to_covariance(
    correlation: Matrix,
    volatility: Vec<f64>,
) -> Result<Matrix, String> {
    let size = validate_square(&correlation)?;
    if volatility.len() != size {
        return Err("volatility length must match correlation dimensions".to_string());
    }
    let mut out = vec![vec![0.0; size]; size];
    for i in 0..size {
        for j in 0..size {
            out[i][j] = correlation[i][j] * volatility[i] * volatility[j];
        }
    }
    Ok(out)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_std(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = mean(values);
    (values
        .iter()
        .map(|value| (value - avg).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

fn sample_corr(left: &[f64], right: &[f64]) -> f64 {
    let left_mean = mean(left);
    let right_mean = mean(right);
    let left_std = sample_std(left);
    let right_std = sample_std(right);
    if left_std == 0.0 || right_std == 0.0 {
        return f64::NAN;
    }
    let cov = left
        .iter()
        .zip(right)
        .map(|(l, r)| (l - left_mean) * (r - right_mean))
        .sum::<f64>()
        / (left.len() - 1) as f64;
    cov / (left_std * right_std)
}

pub fn rolling_correlation(x: Vec<f64>, y: Vec<f64>, window: usize) -> Result<Vec<f64>, String> {
    if window < 2 {
        return Err("window must be >= 2".to_string());
    }
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }
    let mut out = vec![f64::NAN; x.len()];
    for idx in (window - 1)..x.len() {
        out[idx] = sample_corr(&x[idx + 1 - window..idx + 1], &y[idx + 1 - window..idx + 1]);
    }
    Ok(out)
}

pub fn cointegration_spread(y: Vec<f64>, x: Vec<f64>) -> Result<(f64, f64, Vec<f64>), String> {
    if x.is_empty() || x.len() != y.len() {
        return Err("x and y must have the same non-zero length".to_string());
    }
    let x_mean = mean(&x);
    let y_mean = mean(&y);
    let numerator = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f64>();
    let denominator = x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>();
    if denominator == 0.0 {
        return Err("x variance must be positive".to_string());
    }
    let hedge_ratio = numerator / denominator;
    let intercept = y_mean - hedge_ratio * x_mean;
    let raw_spread: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(xi, yi)| yi - (intercept + hedge_ratio * xi))
        .collect();
    let spread_mean = mean(&raw_spread);
    Ok((
        hedge_ratio,
        intercept,
        raw_spread
            .into_iter()
            .map(|value| value - spread_mean)
            .collect(),
    ))
}

fn discretize(values: Vec<f64>, bins: usize) -> Result<Vec<usize>, String> {
    if bins < 2 {
        return Err("bins must be >= 2".to_string());
    }
    if values.is_empty() {
        return Err("values must not be empty".to_string());
    }
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let edges: Vec<f64> = (1..bins)
        .map(|idx| {
            let pos = ((sorted.len() - 1) as f64 * idx as f64 / bins as f64).round() as usize;
            sorted[pos]
        })
        .collect();
    Ok(values
        .into_iter()
        .map(|value| edges.iter().filter(|edge| value >= **edge).count())
        .collect())
}

pub fn transfer_entropy(
    source: Vec<f64>,
    target: Vec<f64>,
    bins: usize,
    lag: usize,
) -> Result<f64, String> {
    if lag < 1 {
        return Err("lag must be >= 1".to_string());
    }
    if source.len() != target.len() {
        return Err("source and target must have the same length".to_string());
    }
    if source.len() <= lag + 1 {
        return Ok(0.0);
    }
    let src = discretize(source, bins)?;
    let tgt = discretize(target, bins)?;
    let future = &tgt[lag..];
    let target_past = &tgt[..tgt.len() - lag];
    let source_past = &src[..src.len() - lag];
    let total = future.len() as f64;
    let mut entropy = 0.0;
    for idx in 0..future.len() {
        let y_next = future[idx];
        let y_prev = target_past[idx];
        let x_prev = source_past[idx];
        let joint_xyz = (0..future.len())
            .filter(|j| {
                future[*j] == y_next && target_past[*j] == y_prev && source_past[*j] == x_prev
            })
            .count() as f64
            / total;
        let joint_yz = (0..future.len())
            .filter(|j| future[*j] == y_next && target_past[*j] == y_prev)
            .count() as f64
            / total;
        let joint_xz = (0..future.len())
            .filter(|j| source_past[*j] == x_prev && target_past[*j] == y_prev)
            .count() as f64
            / total;
        let prob_z = target_past.iter().filter(|value| **value == y_prev).count() as f64 / total;
        if joint_xyz > 0.0 && joint_yz > 0.0 && joint_xz > 0.0 && prob_z > 0.0 {
            entropy += joint_xyz * ((joint_xyz * prob_z) / (joint_yz * joint_xz)).ln();
        }
    }
    Ok(entropy.max(0.0))
}

pub fn twap_schedule(total_quantity: f64, periods: usize) -> Result<Vec<f64>, String> {
    if periods == 0 {
        return Err("periods must be positive".to_string());
    }
    Ok(vec![total_quantity / periods as f64; periods])
}

pub fn vwap_schedule(total_quantity: f64, volume_curve: Vec<f64>) -> Result<Vec<f64>, String> {
    if volume_curve.is_empty() || volume_curve.iter().any(|value| *value < 0.0) {
        return Err("volume_curve must contain positive total volume".to_string());
    }
    let total_volume = volume_curve.iter().sum::<f64>();
    if total_volume <= 0.0 {
        return Err("volume_curve must contain positive total volume".to_string());
    }
    Ok(volume_curve
        .into_iter()
        .map(|value| total_quantity * value / total_volume)
        .collect())
}

pub fn almgren_chriss_market_impact(
    quantity: f64,
    volatility: f64,
    daily_volume: f64,
    spread: f64,
    temporary_impact: f64,
    permanent_impact: f64,
) -> Result<(f64, f64, f64, f64), String> {
    if daily_volume <= 0.0 {
        return Err("daily_volume must be positive".to_string());
    }
    let participation = quantity.abs() / daily_volume;
    let spread_cost = quantity.abs() * spread / 2.0;
    let temp_cost = quantity.abs() * temporary_impact * volatility * participation;
    let perm_cost = quantity.abs() * permanent_impact * volatility * participation.sqrt();
    Ok((
        spread_cost,
        temp_cost,
        perm_cost,
        spread_cost + temp_cost + perm_cost,
    ))
}

pub fn optimal_execution_trajectory(
    total_quantity: f64,
    periods: usize,
    risk_aversion: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if periods == 0 {
        return Err("periods must be positive".to_string());
    }
    if risk_aversion < 0.0 {
        return Err("risk_aversion must be non-negative".to_string());
    }
    let mut remaining = Vec::with_capacity(periods + 1);
    for idx in 0..=periods {
        let time = idx as f64 / periods as f64;
        let value = if risk_aversion > 0.0 {
            let decay = (-risk_aversion * time).exp();
            let end_decay = (-risk_aversion).exp();
            total_quantity * (decay - end_decay) / (1.0 - end_decay)
        } else {
            total_quantity * (1.0 - time)
        };
        remaining.push(value);
    }
    if let Some(last) = remaining.last_mut() {
        *last = 0.0;
    }
    let trades = remaining.windows(2).map(|pair| pair[0] - pair[1]).collect();
    Ok((remaining, trades))
}

pub fn factor_risk_decomposition(
    weights: Vec<f64>,
    factor_loadings: Matrix,
    factor_covariance: Matrix,
    specific_variance: Vec<f64>,
) -> Result<(f64, f64, f64, Vec<f64>), String> {
    let (assets, factors) = validate_matrix(&factor_loadings)?;
    if weights.len() != assets || specific_variance.len() != assets {
        return Err(
            "weights and specific_variance length must match factor_loadings rows".to_string(),
        );
    }
    if validate_square(&factor_covariance)? != factors {
        return Err("factor_covariance dimensions must match factor_loadings columns".to_string());
    }
    let mut exposure = vec![0.0; factors];
    for asset in 0..assets {
        for factor in 0..factors {
            exposure[factor] += weights[asset] * factor_loadings[asset][factor];
        }
    }
    let mut marginal = vec![0.0; factors];
    for i in 0..factors {
        for j in 0..factors {
            marginal[i] += factor_covariance[i][j] * exposure[j];
        }
    }
    let contributions: Vec<f64> = exposure
        .iter()
        .zip(&marginal)
        .map(|(exp, marg)| exp * marg)
        .collect();
    let systematic = contributions.iter().sum::<f64>();
    let specific = weights
        .iter()
        .zip(&specific_variance)
        .map(|(weight, variance)| weight.powi(2) * variance)
        .sum::<f64>();
    Ok((systematic + specific, systematic, specific, contributions))
}

/**********************************/
#[cfg(test)]
mod example_tests {
    #[test]
    fn test_placeholder() {
        assert_eq!(2 + 2, 4);
    }
}
