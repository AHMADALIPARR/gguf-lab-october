// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! Exact arithmetic in Z[phi], phi = (1+sqrt 5)/2, phi^2 = phi + 1.
//! Port of the checkable parts of sovereign-ruby-kernel lib/ast/phi_algebra.rb
//! (literals, phi^k, add, mul, Galois conjugate, norm), which the Ruby code
//! evaluates in f64. Here everything is i128 with checked operations; there
//! is no floating point anywhere in this module. Overflow is an error.
//!
//! Element a + b*phi is stored as (a, b).
//!   phi^k            = F(k-1) + F(k) phi            (k >= 1), phi^0 = 1
//!   (a+b phi)(c+d phi) = (ac + bd) + (ad + bc + bd) phi
//!   conj(a+b phi)    = (a+b) - b phi                (phi -> phi_hat = 1 - phi)
//!   norm(a+b phi)    = (a+b phi)(conj)  = a^2 + ab - b^2   (an integer)

pub type Z = i128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZPhi { pub a: Z, pub b: Z }

fn ck(x: Option<Z>) -> Result<Z, String> { x.ok_or_else(|| "integer overflow".to_string()) }

impl ZPhi {
    pub fn add(self, o: ZPhi) -> Result<ZPhi, String> {
        Ok(ZPhi { a: ck(self.a.checked_add(o.a))?, b: ck(self.b.checked_add(o.b))? })
    }
    pub fn mul(self, o: ZPhi) -> Result<ZPhi, String> {
        let ac = ck(self.a.checked_mul(o.a))?;
        let bd = ck(self.b.checked_mul(o.b))?;
        let ad = ck(self.a.checked_mul(o.b))?;
        let bc = ck(self.b.checked_mul(o.a))?;
        Ok(ZPhi { a: ck(ac.checked_add(bd))?, b: ck(ck(ad.checked_add(bc))?.checked_add(bd))? })
    }
    pub fn scale(self, c: Z) -> Result<ZPhi, String> {
        Ok(ZPhi { a: ck(self.a.checked_mul(c))?, b: ck(self.b.checked_mul(c))? })
    }
    pub fn conj(self) -> Result<ZPhi, String> {
        Ok(ZPhi { a: ck(self.a.checked_add(self.b))?, b: ck(self.b.checked_neg())? })
    }
    pub fn norm(self) -> Result<Z, String> {
        let aa = ck(self.a.checked_mul(self.a))?;
        let ab = ck(self.a.checked_mul(self.b))?;
        let bb = ck(self.b.checked_mul(self.b))?;
        ck(ck(aa.checked_add(ab))?.checked_sub(bb))
    }
    /// Canonical text: "a + b phi" / "a - |b| phi".
    pub fn fmt(self) -> String {
        if self.b < 0 { format!("{} - {} phi", self.a, -self.b) } else { format!("{} + {} phi", self.a, self.b) }
    }
}

pub fn phi_pow(k: u32) -> Result<ZPhi, String> {
    if k == 0 { return Ok(ZPhi { a: 1, b: 0 }); }
    let (mut f0, mut f1): (Z, Z) = (0, 1); // F(k-1), F(k) for k = 1
    for _ in 1..k {
        let f2 = ck(f0.checked_add(f1))?;
        f0 = f1; f1 = f2;
    }
    Ok(ZPhi { a: f0, b: f1 })
}

/// Sum of c_i * phi^k_i.
pub fn normalize(terms: &[(Z, u32)]) -> Result<ZPhi, String> {
    let mut acc = ZPhi { a: 0, b: 0 };
    for &(c, k) in terms { acc = acc.add(phi_pow(k)?.scale(c)?)?; }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identities() {
        assert_eq!(phi_pow(2).unwrap(), ZPhi { a: 1, b: 1 });       // phi^2 = 1 + phi
        assert_eq!(phi_pow(5).unwrap(), ZPhi { a: 3, b: 5 });
        let x = ZPhi { a: 3, b: -2 };
        assert_eq!(x.mul(x.conj().unwrap()).unwrap(), ZPhi { a: x.norm().unwrap(), b: 0 });
        // README canonical TRS * 5 = 534 + 872 phi; norm(TRS) = -1916/5
        assert_eq!(ZPhi { a: 534, b: 872 }.norm().unwrap(), -1916 * 5);
    }
}
