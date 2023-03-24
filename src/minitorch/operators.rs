use std::iter::Iterator;

pub fn mul(x: f64, y: f64) -> f64 {
    x * y
}

pub fn id(x: f64) -> f64  {
    x
}

pub fn add(x: f64, y: f64) -> f64 {
    x + y
}

pub fn neg(x: f64) -> f64 {
    -x
}

pub fn lt(x: f64, y:f64) -> f64 {
    (x < y) as i64 as f64
}

pub fn eq(x: f64, y: f64) -> f64 {
    (x == y) as i64 as f64
}

pub fn max(x: f64, y: f64) -> f64 {
    f64::max(y, x)
}

pub fn inv(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    1.0 / x
}

pub fn inv_back(x: f64, d: f64) -> f64 {
    return -d/mul(x, x)
}

pub fn is_close(x: f64, y: f64) -> f64 {
    (f64::abs(x-y) < 1e-2) as i64 as f64
}

pub fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

pub fn relu_back(x: f64, d: f64) -> f64 {
    if x > 0.0 {
        return d;
    }
    return 0.0
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        return 1.0 / (1.0 + f64::exp(-x))
    }
    f64::exp(x)/(1.0 + f64::exp(x))
}

pub fn log(x: f64) -> f64 {
    if x == 0.0 {
        return 1e-6_f64.ln();
    }
    x.ln()
}

pub fn log_back(x: f64, d: f64) -> f64 {
    mul(inv(x), d)
}

pub fn exp(x: f64) -> f64 {
    f64::exp(x)
}

pub fn map<F>(func: F) -> impl Fn(Vec<f64>) -> Vec<f64> 
where 
    F: Fn(f64) -> f64
{
    move |iterable: Vec<f64>| -> Vec<f64> {
        iterable.into_iter().map(|value| func(value)).collect::<Vec<f64>>()
    }
}


// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_close;
    use crate::test_utils::gen_float;
    use rstest::rstest;

    #[rstest]
    fn test_basic() {
        assert_close(mul(1.0, 2.0), 2.0);
        assert_close(add(1.0, 2.0), 3.0);
        assert_close(neg(1.0), -1.0);
        assert_close(max(0.0, 1.0), 1.0);
        assert_close(max(1.0, 1.0), 1.0);
        assert_close(inv(2.0), 0.5);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_relu(#[case] input: f64) {
        if input > 0.0 {
            assert!(relu(input) == input);
        } else {
            assert!(relu(input) == 0.0);
        }
    }

    #[rstest]
    #[case(gen_float())]
    fn test_id(#[case] input: f64) {
        assert!(id(input) == input);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_lt(#[case] input: f64) {
        assert!(lt(input, input - 1.0) == 0.0);
        assert!(lt(input - 1.0, input) == 1.0);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_max(#[case] input: f64) {
        assert!(max(input, input - 1.0) == input);
        assert!(max(input - 1.0, input) == input);
        assert!(max(input, input) == input);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_eq(#[case] input: f64) {
        assert!(eq(input, input) == 1.0);
        assert!(eq(input, input + 1.0) == 0.0);
        assert!(eq(input + 1.0, input) == 0.0);
    }

    #[rstest]
    #[case(gen_float(), gen_float())]
    fn test_relu_back(#[case] input: f64, #[case] constant: f64) {
        if input > 0.0 {
            assert!(relu_back(input, constant) == constant);
        } else if input < 0.0 {
            assert!(relu_back(input, constant) == 0.0);
        }
    }

    #[rstest]
    #[case(gen_float())]
    fn test_sigmoid(#[case] input: f64) {
        // Test sigmoid properties
        assert!(0.0 <= sigmoid(input) && sigmoid(input) <= 1.0);
        assert_close(1.0 - sigmoid(input), sigmoid(-input));
        assert_close(sigmoid(0.0), 0.5);
        assert!(
            sigmoid(input + 1.0) >= sigmoid(input) 
            && sigmoid(input) >= sigmoid(input - 1.0)
            && sigmoid(input - 1.0)>= 0.0
        );
    }

    #[rstest]
    #[case(gen_float(), gen_float(), gen_float())]
    fn test_transitive(#[case] input_1: f64, #[case] input_2: f64, #[case] input_3: f64) {
        /* Test the transitive property -- if a<b and b<c then a<c */
        let mut sorted = vec![input_1, input_2, input_3];
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let a = sorted[0];
        let b = sorted[1];
        let c = sorted[2];
        if lt(a, b) + lt(b, c) == 2.0 {
            assert!(lt(a, c) == 1.0);
        }
    }

    #[rstest]
    #[case(gen_float(), gen_float())]
    fn test_commutative(#[case] input_1: f64, #[case] input_2: f64) {
        /* Test the commutative property of addition and multiplication */
        assert_close(add(input_1, input_2), add(input_2, input_1));
        assert_close(mul(input_1, input_2), mul(input_2, input_1));
    }

    #[rstest]
    #[case(gen_float(), gen_float(), gen_float())]
    fn test_distributive(#[case] input_1: f64, #[case] input_2: f64, #[case] input_3: f64) {
        /* Test the distributive property of multiplication over addition */
        assert_close(mul(input_1, add(input_2, input_3)), add(mul(input_1, input_2), mul(input_1, input_3)));
    }

    #[rstest]
    #[case(gen_float(), gen_float())]
    fn test_map_id(#[case] input_1: f64, #[case] input_2: f64) {
        let map_id = map(id);
        let v = vec![input_1, input_2];
        assert!(map_id(v.clone()) == v);
    }
}
