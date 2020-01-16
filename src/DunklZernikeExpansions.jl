module DunklZernikeExpansions

import Base: +, -, *, /, ==, isapprox
import Jacobi:jacobi
import SpecialFunctions:gamma

export DZFun, DZParam, DZPoly, evalDZ, mbx1, mbx2, symx1, symx2, skewx1, skewx2, Dunklx1, Dunklx2, DunklAngular, project, mbr, adjointDunklx1, adjointDunklx2

function inferDegree(l::Int64)
	# Given l it returns two integers; the first one is the lowest integer n such that (n+1)(n+2)÷2 ≥ l;
	# the second one is the residual (n+1)(n+2)÷2 - l
	n = (-3 + sqrt(1+8*l))/2 # This will be a float
	cn = convert(Int64, ceil(n))
	cn, (cn+1)*(cn+2)÷2-l
end

polyDim(deg::Int64) = (deg+1)*(deg+2)÷2    

struct DZParam
	γ1::Float64
	γ2::Float64
	α::Float64
	function DZParam(γ1, γ2, α)
		@assert γ1 > -1 && γ2 > -1 && α > -1
		new(γ1, γ2, α)
	end
end

function isapprox(κ1::DZParam, κ2::DZParam)
	a = 1.0e-12
	isapprox(κ1.γ1,κ2.γ1;atol=a) && isapprox(κ1.γ2,κ2.γ2;atol=a) && isapprox(κ1.α,κ2.α;atol=a)
end

struct DZFun
	κ::DZParam
	degree::Int64
	coefficients::Vector{Float64}
	# The coefficients of a polynomial are ordered by degree first;
	# within each degree, the coefficients accompanying generalized cosines appear in the odd-indexed positions and generalized sines in the even-indexed positions;
	# within the odd (resp. even) positions the coefficients accompanying Dunkl–Zernike polynomials involving spherical harmonics of higher degree appear first
	function DZFun(κ, degree, coefficients)
		@assert 2*length(coefficients) == (degree+1)*(degree+2)
		new(κ, degree, coefficients)
	end
end

function DZFun(κ::Tuple{T1,T2,T3}, degree::Int64, coefficients::Vector{Float64}) where {T1<:Real, T2<:Real, T3<:Real}
	param = DZParam(κ...)
	DZFun(param, degree, coefficients)
end

function DZFun(κ::Vector{T}, degree::Int64, coefficients::Vector{Float64}) where T<:Real
	@assert length(κ) == 3 "If the parameter is given as a vector it must be of length 3"
	param = DZParam(κ...)
	DZFun(param, degree, coefficients)
end

function DZFun(κ, coefficients::Vector{T}) where T<:Real
	n, res = inferDegree(length(coefficients))
	cl = polyDim(n)
	newcoefficients = zeros(Float64, cl)
	newcoefficients[1:length(coefficients)] = coefficients
	DZFun(κ, n, newcoefficients)
end

function ==(f::DZFun, g::DZFun)
	equalκ = f.κ == g.κ
	fl = length(f.coefficients)
	gl = length(g.coefficients)
	maxl = max(fl,gl)
	equalc = [f.coefficients;zeros(maxl-fl)] == [g.coefficients;zeros(maxl-gl)]
	equalκ && equalc
end

function isapprox(f::DZFun, g::DZFun)
	equalκ = f.κ ≈ g.κ
	fl = length(f.coefficients)
	gl = length(g.coefficients)
	maxl = max(fl,gl)
	equalc = [f.coefficients;zeros(maxl-fl)] ≈ [g.coefficients;zeros(maxl-gl)]
	equalκ && equalc
end

# Unary operations
-(f::DZFun) = DZFun(f.κ, f.degree, -f.coefficients)

# Binary operations
for op = (:+, :-)
	@eval begin
		function ($op)(f::DZFun, g::DZFun)
			@assert f.κ ≈ g.κ
			fl = length(f.coefficients)
			gl = length(g.coefficients)
			retl = max(fl, gl)
			retd = max(f.degree, g.degree)
			retcoefficients = zeros(Float64, retl)
			retcoefficients[1:fl] = f.coefficients;
			retcoefficients[1:gl] = ($op)(retcoefficients[1:gl], g.coefficients);
			DZFun(f.κ, retd, retcoefficients)
		end
	end
end

# Operations with scalars
for op = (:+, :-)
	@eval begin
		function ($op)(f::DZFun, a::Number)
			($op)(f, DZFun(f.κ, 0, [a]))
		end
	end
end
for op = (:*, :/)
	@eval begin
		function ($op)(f::DZFun, a::Number)
			DZFun(f.κ, f.degree, ($op)(f.coefficients, a))
		end
	end
end
for op = (:+, :*)
	@eval begin
		($op)(a::Number, f::DZFun) = ($op)(f, a)
	end
end
-(a::Number, f::DZFun) = a + (-f)

# Position range of coefficients of given degree
positionRange(deg::Integer) = (polyDim(deg-1)+1):polyDim(deg)

function pairing(m::Int64, n::Int64, even::Bool)
	@assert m≥0 && n≥0
	@assert m>0 || even
	deg = m+2n
	1+polyDim(deg-1)+(~even)+2*n
end

function inversepairing(i::Int64)
	@assert i≥0
	deg, res = inferDegree(i)
	n = (deg-res)÷2
	m = deg-2*n
	even = ~Bool((deg-res)%2)
	(m,n,even)
end

# Dunkl–Zernike polynomials
function DZPoly(κ::DZParam, m::Int64, n::Int64, even::Bool)
	i = pairing(m, n, even)
	v = zeros(Float64, i)
	v[i] = 1.0
	DZFun(κ, v)
end

function DZPoly(κ::Tuple{T1,T2,T3}, m::Int64, n::Int64, even::Bool) where {T1<:Real, T2<:Real, T3<:Real}
	param = DZParam(κ...)
	DZPoly(param, m, n, even)
end

function DZPoly(κ::Vector{T}, m::Int64, n::Int64, even::Bool) where T<:Real
	@assert length(κ) == 3 "If the parameter is given as a vector it must be of length 3"
	param = DZParam(κ...)
	DZPoly(param, m, n, even)
end

function project(f::DZFun, N::Int64)
	if f.degree≤N
		f
	else
		DZFun(f.κ,f.coefficients[1:polyDim(N)])
	end
end

"""
Express a DZFun in a base with α raised by 1
"""
function raise(f::DZFun)
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	α = f.κ.α
	N = f.degree
	outκ = DZParam(γ1, γ2, α+1)
	outcoefs = zeros(Float64, length(f.coefficients))
	for n = 0:N÷2
		poscos = pairing(0,n,true)
		if n == 0
			outcoefs[poscos] = f.coefficients[poscos]
		else
			outcoefs[poscos] = (n+α+(γ1+γ2)/2+1)/(2n+α+(γ1+γ2)/2+1)*f.coefficients[poscos]
		end
		if 2n≤N-2
			poscosup = pairing(0,n+1,true)
			outcoefs[poscos] -= (n+1+(γ1+γ2)/2)/(2n+α+3+(γ1+γ2)/2)*f.coefficients[poscosup]
		end
		for m = 1:N - 2n
			poscos = pairing(m,n,true)
			possin = pairing(m,n,false)
			outcoefs[poscos] = (m+n+α+(γ1+γ2)/2+1)/(m+2n+α+(γ1+γ2)/2+1)*f.coefficients[poscos]
			outcoefs[possin] = (m+n+α+(γ1+γ2)/2+1)/(m+2n+α+(γ1+γ2)/2+1)*f.coefficients[possin]
			if m+2n≤N-2
				poscosup = pairing(m,n+1,true)
				possinup = pairing(m,n+1,false)
				outcoefs[poscos] -= (m+n+1+(γ1+γ2)/2)/(m+2n+α+3+(γ1+γ2)/2)*f.coefficients[poscosup]
				outcoefs[possin] -= (m+n+1+(γ1+γ2)/2)/(m+2n+α+3+(γ1+γ2)/2)*f.coefficients[possinup]
			end
		end
	end
	DZFun(outκ, N, outcoefs)
end

"""
Express a DZFun in a base with α lowered by 1
"""
function lower(f::DZFun)
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	α = f.κ.α - 1 #New α
	@assert α>-1
	N = f.degree
	outκ = DZParam(γ1, γ2, α)
	origcoefs = f.coefficients
	outcoefs = zeros(Float64, length(origcoefs))
	for n = N÷2:-1:0
		m = 0	
		poscos = pairing(m,n,true)
		if n == 0
			if m+2n>N-2
				outcoefs[poscos] = origcoefs[poscos]
			else		
				poscosup = pairing(m,n+1,true)
				outcoefs[poscos] = origcoefs[poscos] + (m+n+1+(γ1+γ2)/2)/(m+2n+3+α+(γ1+γ2)/2)*outcoefs[poscosup]
			end
		else
			if m+2n>N-2
				outcoefs[poscos] = (m+2n+α+(γ1+γ2)/2+1)/(m+n+α+(γ1+γ2)/2+1)*origcoefs[poscos]			
			else
				poscosup = pairing(m,n+1,true)
				outcoefs[poscos] = (m+2n+α+(γ1+γ2)/2+1)/(m+n+α+(γ1+γ2)/2+1)*( origcoefs[poscos] + (m+n+1+(γ1+γ2)/2)/(m+2n+3+α+(γ1+γ2)/2)*outcoefs[poscosup] )
			end
		end
		for m = 1:N-2n
			if m+2n>N-2
				poscos = pairing(m,n,true)
				possin = pairing(m,n,false)
				outcoefs[poscos] = (m+2n+α+(γ1+γ2)/2+1)/(m+n+α+(γ1+γ2)/2+1)*origcoefs[poscos]
				outcoefs[possin] = (m+2n+α+(γ1+γ2)/2+1)/(m+n+α+(γ1+γ2)/2+1)*origcoefs[possin]
			else
				poscos = pairing(m,n,true)
				possin = pairing(m,n,false)
				poscosup = pairing(m,n+1,true)
				possinup = pairing(m,n+1,false)
				outcoefs[poscos] = (m+2n+α+(γ1+γ2)/2+1)/(m+n+α+(γ1+γ2)/2+1)*( origcoefs[poscos] + (m+n+1+(γ1+γ2)/2)/(m+2n+3+α+(γ1+γ2)/2)*outcoefs[poscosup] )
				outcoefs[possin] = (m+2n+α+(γ1+γ2)/2+1)/(m+n+α+(γ1+γ2)/2+1)*( origcoefs[possin] + (m+n+1+(γ1+γ2)/2)/(m+2n+3+α+(γ1+γ2)/2)*outcoefs[possinup] )
			end
		end
	end
	DZFun(outκ, N, outcoefs)
end

"""
Evaluate Generalized Gegenbauer
"""
function genGeg(x::Number,n::Integer,lam::Number,mu::Number)
	if iseven(n)
		return jacobi(2x^2-1,n÷2,lam-0.5,mu-0.5)
	else
		return x*jacobi(2x^2-1,(n-1)÷2,lam-0.5,mu+0.5)
	end
end

"""
Square norm of a Jacobi polynomial
"""
function jacsqn(n::Integer,α::Float64,β::Float64)
	if n == 0 && α+β+1≈0
		2^(α+β+1)*gamma(α+1)*gamma(β+1)
	else
		(2^(α+β+1)/(2n+α+β+1))*( (gamma(n+α+1)*gamma(n+β+1))/(gamma(n+α+β+1)*factorial(n)) )
	end
end

"""
Square norm of a Generalized Gegenbauer polynomial
"""
function ggsqn(n::Integer,lam::Number,mu::Number)
	if iseven(n)
		jacsqn(n÷2,lam-0.5,mu-0.5)/2^(lam+mu)
	else
		jacsqn((n-1)÷2,lam-0.5,mu+0.5)/2^(lam+mu+1)
	end
end

"""
Square norm (on the circle) of hharmonic
"""
function hhsqn(m::Integer,γ1::Float64,γ2::Float64,even::Bool)
	if even
		2*ggsqn(m,γ2/2,γ1/2)
	else
		2*ggsqn(m-1,γ2/2+1,γ1/2)
	end
end

"""
Square norm of an element of a Dunkl-Zernike polynomial
"""
function DZsqn(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64,even::Bool)
	jacsqn(n,α,m+(γ1+γ2)/2)/2^(m+α+(γ1+γ2)/2+2)*hhsqn(m,γ1,γ2,even)
end

"""
Compute inner product between two DZFun with the same parameters
"""
function DZFunInner(f::DZFun,g::DZFun)
	@assert f.κ ≈ g.κ
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	α = f.κ.α
	vf = f.coefficients
	vg = g.coefficients

	l = min(length(vf),length(vg))
	out = 0.0
	for j=1:l
		(m,n,even) = inversepairing(j)
		out += vf[j]*vg[j]*DZsqn(m,n,α,γ1,γ2,even)
	end
	out
end

#########################################################################################################
"""
Compute the ratio between the weighted square norms of two consecutive Jacobi polynomials of same parameters
"""
JacDegreeRatio(n::Integer,α::Float64,β::Float64) = ((2n+α+β+1)/(2n+α+β+3))*((n+α+1)/(n+α+β+1))*((n+β+1)/(n+1))

"""
Compute the ratio between twe weighted square norm of two Jacobi polynoamials of same degree and first parameter but differing in its second parameter in two units
"""
JacParameterRatio(n::Integer,α::Float64,β::Float64) = 4*((2n+α+β+1)/(2n+α+β+3))*((n+β+2)/(n+α+β+2))*((n+β+1)/(n+α+β+1))

"""
Compute the ratio between the weighted square norm of two Generalized Gegenbauer polynomials of same parameters but differing in the degree in two units.
"""
function GGRatio(n::Integer,λ::Float64,μ::Float64)
	if iseven(n)
		JacDegreeRatio(n÷2,λ-0.5,μ-0.5)
	else
		JacDegreeRatio((n-1)÷2,λ-0.5,μ+0.5)
	end
end

"""
Compute the ratio between the weighted square norm of two h-harmonic polynomials of same parameters but differing in the degree in two units
"""
function hhRatio(m::Integer,γ1::Float64,γ2::Float64,even::Bool)
	if even
		GGRatio(m,γ2/2,γ1/2)
	else
		GGRatio(m-1,γ2/2+1,γ1/2)
	end
end

"""
Compute the ratio between the weighted square norm of two DZ polynomials of same parameters but differing in n in one unit
"""
DZnRatio(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64,even::Bool) = JacDegreeRatio(n,α,m+(γ1+γ2)/2)

"""
Compute the ratio between the weighted square norm of two DZ polynomials of same parameters but differing in m in two units
"""
DZmRatio(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64,even::Bool) = .5*JacParameterRatio(n,α,m+(γ1+γ2)/2)*hhRatio(m,γ1,γ2,even)
#########################################################################################################

"""
Evaluate DZFun
"""
function evalDZ(f::DZFun,x::Number,y::Number)
	out = 0.0
	coefficients = f.coefficients
	α = f.κ.α
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	r2 = x^2+y^2
	t = atan(y,x)
	for j = 1:length(coefficients)
		(m,n,even) = inversepairing(j)
		if even
			out += coefficients[j]*r2^(m/2)*genGeg(cos(t),m,γ2/2,γ1/2)*jacobi(2r2-1,n,α,m+(γ1+γ2)/2)
		else
			out += coefficients[j]*r2^(m/2)*sin(t)*genGeg(cos(t),m-1,γ2/2+1,γ1/2)*jacobi(2r2-1,n,α,m+(γ1+γ2)/2)
		end
	end
	out
end

function D1even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		(m+γ2-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	else
		(m+γ1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	end
end
function E1even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		(m+γ1+γ2)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	else
		(m+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	end
end
function D2even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		-(m+γ1-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	else
		-(m+γ1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	end
end
function E2even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		(m+γ1+γ2)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	else
		(m+γ1+γ2+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	end
end
function D1odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		(m+γ1-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	else
		(m+γ2)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	end
end
function E1odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		m*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	else
		(m+γ1+γ2+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	end
end
function D2odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		(m+γ2-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	else
		(m+γ2)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)
	end
end
function E2odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if iseven(m)
		-m*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	else
		-(m+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)
	end
end
F1even(n::Integer,α::Float64,γ1::Float64,γ2::Float64) = 2n+2α+γ1+γ2+2
F2even(n::Integer,α::Float64,γ1::Float64,γ2::Float64) = 2n+2α+γ1+γ2+2

"""
Dunkl-x1 operator with shift
"""
function DunklShiftx1(f::DZFun)
	OrigCoeff = f.coefficients
	α = f.κ.α
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	N = f.degree

	OutDegree = max(0,N-1)
	OutCoeff = zeros(polyDim(OutDegree))
	
	m = 0
	for n=0:fld(N-1,2)
		ixMN = pairing(m,n,true) # Index associated to (0,n,Even)
		ixMpN = pairing(m+1,n,true) # Index associated to (1,n,Even)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*D1even(m+1,n,α,γ1,γ2)
	end

	m = 1
	for n=0:fld(N-1-m,2)
		ixMN = pairing(m,n,true) # Index associated to (1,n,Even)
		ixMpN = pairing(m+1,n,true) # Index associated to (2,n,Even)
		ixMmNp = pairing(m-1,n+1,true) # Index associated to (0,n+1,Even)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*D1even(m+1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*F1even(n+1,α,γ1,γ2)

		ixMN = pairing(m,n,false) # Index associated to (1,n,Odd)
		ixMpN = pairing(m+1,n,false) # Index associated to (2,n,Odd)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*D1odd(m+1,n,α,γ1,γ2)
	end
	for m=2:(N-1)
		for n=0:(N-1-m)÷2
			ixMN = pairing(m,n,true) # Index associated to (m,n,Even)
			ixMpN = pairing(m+1,n,true) # Index associated to (m+1,n,Even)
			ixMmNp = pairing(m-1,n+1,true) # Index associated to (m-1,n+1,Even)
			OutCoeff[ixMN] = OrigCoeff[ixMpN]*D1even(m+1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*E1even(m-1,n+1,α,γ1,γ2)

			ixMN = pairing(m,n,false) # Index associated to (m,n,Odd)
			ixMpN = pairing(m+1,n,false) # Index associated to (m+1,n,Odd)
			ixMmNp = pairing(m-1,n+1,false) # Index associated to (m-1,n+1,Odd)
			OutCoeff[ixMN] = OrigCoeff[ixMpN]*D1odd(m+1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*E1odd(m-1,n+1,α,γ1,γ2)
		end
	end
	DZFun([γ1,γ2,α+1],OutDegree,OutCoeff)
end

"""
Dunkl-x2 operator with shift
"""
function DunklShiftx2(f::DZFun)
	OrigCoeff = f.coefficients
	α = f.κ.α
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	N = f.degree

	OutDegree = max(0,N-1)
	OutCoeff = zeros(polyDim(OutDegree))
	
	m = 0
	for n=0:fld(N-1,2)
		ixMN = pairing(m,n,true) # Index associated to (0,n,Even)
		ixMpN = pairing(m+1,n,false) # Index associated to (1,n,Odd)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*D2odd(m+1,n,α,γ1,γ2)
	end

	m = 1
	for n=0:fld(N-1-m,2)
		ixMN = pairing(m,n,true) # Index associated to (1,n,Even)
		ixMpN = pairing(m+1,n,false) # Index associated to (2,n,odd)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*D2odd(m+1,n,α,γ1,γ2)

		ixMN = pairing(m,n,false) # Index associated to (1,n,Odd)
		ixMpN = pairing(m+1,n,true) # Index associated to (2,n,Even)
		ixMmNp = pairing(m-1,n+1,true) # Index assoaciated to (0,n+1,Even)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*D2even(m+1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*F2even(n+1,α,γ1,γ2)
	end
	for m=2:(N-1)
		for n=0:(N-1-m)÷2
			ixMN = pairing(m,n,true) # Index associated to (m,n,Even)
			ixMpN = pairing(m+1,n,false) # Index associated to (m+1,n,Odd)
			ixMmNp = pairing(m-1,n+1,false) # Index associated to (m-1,n+1,Odd)
			OutCoeff[ixMN] = OrigCoeff[ixMpN]*D2odd(m+1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*E2odd(m-1,n+1,α,γ1,γ2)

			ixMN = pairing(m,n,false) # Index associated to (m,n,Odd)
			ixMpN = pairing(m+1,n,true) # Index associated to (m+1,n,Even)
			ixMmNp = pairing(m-1,n+1,true) # Index associated to (m-1,n+1,Even)
			OutCoeff[ixMN] = OrigCoeff[ixMpN]*D2even(m+1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*E2even(m-1,n+1,α,γ1,γ2)
		end
	end
	DZFun([γ1,γ2,α+1],OutDegree,OutCoeff)
end

"""
Unshifted Dunkl operators
"""
Dunklx1(f::DZFun) = lower(DunklShiftx1(f))
Dunklx2(f::DZFun) = lower(DunklShiftx2(f))

function G1even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	elseif m > 0
		(m+γ1+γ2)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	elseif n > 0
		(2m+2n+2α+γ1+γ2+2)/(2m+4n+2α+γ1+γ2+2)
	else
		1.0
	end
end

function H1even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+1)*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	elseif m > 0
		(m+γ1+γ2)*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(2n+2α)/(2m+4n+2α+γ1+γ2+2)
	end
end

function I1even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(m+γ2-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function J1even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ1)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(m+γ2-1)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function G2even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ1+γ2+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	elseif m > 0
		(m+γ1+γ2)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	elseif n > 0
		(2m+2n+2α+γ1+γ2+2)/(2m+4n+2α+γ1+γ2+2)
	else
		1.0
	end
end

function H2even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ1+γ2+1)*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	elseif m > 0
		(m+γ1+γ2)*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(2n+2α)/(2m+4n+2α+γ1+γ2+2)
	end
end

function I2even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		-(m+γ1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		-(m+γ1-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function J2even(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		-(m+γ1)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		-(m+γ1-1)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function G1odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ1+γ2+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		m*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function H1odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ1+γ2+1)*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		m*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function I1odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ2)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(m+γ1-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function J1odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ2)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(m+γ1-1)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function G2odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		-(m+1)*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		-m*(2m+2n+2α+γ1+γ2+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function H2odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		-(m+1)*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		-m*(2n+2α)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function I2odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ2)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(m+γ2-1)*(2m+2n+γ1+γ2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

function J2odd(m::Integer,n::Integer,α::Float64,γ1::Float64,γ2::Float64)
	if isodd(m)
		(m+γ2)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	else
		(m+γ2-1)*(2n+2)/(2m+γ1+γ2)/(2m+4n+2α+γ1+γ2+2)
	end
end

"""
Compute the result of multiplying a DZFun by x1
"""
function mbx1(f::DZFun)
	OrigCoeff = f.coefficients
	α = f.κ.α
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	N = f.degree

	OutCoeff = zeros(polyDim(N+1))

	# Even part

	m = 0
	n = 0
	ixMN = pairing(m,n,true) # Index associated to (0,0,Even)
	if m+2n≤N-1
		ixMpN = pairing(m+1,n,true) # Index assoacited to (1,0,Even)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*I1even(m+1,n,α,γ1,γ2)
	else
		OutCoeff[ixMN] = 0
	end

	n = 0
	for m = 1:N+1-2n
		ixMN = pairing(m,n,true) # Index associated to (m,0,Even)
		ixMmN = pairing(m-1,n,true) # Index associated to (m-1,0,Even)
		if m+2n≤N-1
			ixMmNp = pairing(m-1,n+1,true) # Index associated to (m-1,1,Even)
			ixMpN = pairing(m+1,n,true) # Index associated to (m+1,n,Even)
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1even(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H1even(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I1even(m+1,n,α,γ1,γ2)
		else
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1even(m-1,n,α,γ1,γ2)
		end
	end

	m = 0
	for n = 1:(N+1-m)÷2
		ixMN = pairing(m,n,true) # Index associated to (0,n,Even)
		ixMpNm = pairing(m+1,n-1,true) # Index associated to (1,n-1,Even)
		if m+2n≤N-1
			ixMpN = pairing(m+1,n,true) # Index associated to (1,n,Even)
			OutCoeff[ixMN] = OrigCoeff[ixMpN]*I1even(m+1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J1even(m+1,n-1,α,γ1,γ2)
		else
			OutCoeff[ixMN] = OrigCoeff[ixMpNm]*J1even(m+1,n-1,α,γ1,γ2)
		end
	end

	for n = 1:(N+1)÷2
		for m = 1:N+1-2n
			ixMN = pairing(m,n,true) # Index associated to (m,n,Even)
			ixMmN = pairing(m-1,n,true) # Index associated to (m-1,n,Even)
			ixMpNm = pairing(m+1,n-1,true) # Index associated to (m+1,n-1,Even)
			if m+2n≤N-1
				ixMmNp = pairing(m-1,n+1,true) # Index associated to (m-1,n+1,Even)
				ixMpN = pairing(m+1,n,true) # Index associated to (m+1,n,Even)
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1even(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H1even(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I1even(m+1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J1even(m+1,n-1,α,γ1,γ2)
			else
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1even(m-1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J1even(m+1,n-1,α,γ1,γ2)
			end
		end
	end

	# Odd part

	m = 1
	n = 0
	ixMN = pairing(m,n,false) # Index associated to (1,0,Odd)
	if m+2n≤N-1
		ixMpN = pairing(m+1,n,false) # Index assoacited to (2,0,Odd)
		OutCoeff[ixMN] = OrigCoeff[ixMpN]*I1odd(m+1,n,α,γ1,γ2)
	else
		OutCoeff[ixMN] = 0
	end

	n = 0
	for m = 2:N+1-2n
		ixMN = pairing(m,n,false) # Index associated to (m,0,Odd)
		ixMmN = pairing(m-1,n,false) # Index associated to (m-1,0,Odd)
		if m+2n≤N-1
			ixMmNp = pairing(m-1,n+1,false) # Index associated to (m-1,1,Odd)
			ixMpN = pairing(m+1,n,false) # Index associated to (m+1,n,Odd)
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1odd(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H1odd(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I1odd(m+1,n,α,γ1,γ2)
		else
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1odd(m-1,n,α,γ1,γ2)
		end
	end

	m = 1
	for n = 1:(N+1-m)÷2
		ixMN = pairing(m,n,false) # Index associated to (1,n,Odd)
		ixMpNm = pairing(m+1,n-1,false) # Index associated to (2,n-1,Odd)
		if m+2n≤N-1
			ixMpN = pairing(m+1,n,false) # Index associated to (2,n,Odd)
			OutCoeff[ixMN] = OrigCoeff[ixMpN]*I1odd(m+1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J1odd(m+1,n-1,α,γ1,γ2)
		else
			OutCoeff[ixMN] = OrigCoeff[ixMpNm]*J1odd(m+1,n-1,α,γ1,γ2)
		end
	end

	for n = 1:(N+1)÷2
		for m = 2:N+1-2n
			ixMN = pairing(m,n,false) # Index associated to (m,n,Odd)
			ixMmN = pairing(m-1,n,false) # Index associated to (m-1,n,Odd)
			ixMpNm = pairing(m+1,n-1,false) # Index associated to (m+1,n-1,Odd)
			if m+2n≤N-1
				ixMmNp = pairing(m-1,n+1,false) # Index associated to (m-1,n+1,Odd)
				ixMpN = pairing(m+1,n,false) # Index associated to (m+1,n,Odd)
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1odd(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H1odd(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I1odd(m+1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J1odd(m+1,n-1,α,γ1,γ2)
			else
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G1odd(m-1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J1odd(m+1,n-1,α,γ1,γ2)
			end
		end
	end
	DZFun([γ1,γ2,α],N+1,OutCoeff)
end

"""
Compute the result of multiplying a DZFun by x2
"""
function mbx2(f::DZFun)
	OrigCoeff = f.coefficients
	α = f.κ.α
	γ1 = f.κ.γ1
	γ2 = f.κ.γ2
	N = f.degree

	OutCoeff = zeros(polyDim(N+1))

	# Even part

	n = 0
	for m = 0:1
		ixMN = pairing(m,n,true) # Index associated to (m,0,Even)
		if m+2n≤N-1
			ixMpN = pairing(m+1,n,false) # Index associated to (m+1,0,Odd)
			OutCoeff[ixMN] = OrigCoeff[ixMpN]*I2odd(m+1,n,α,γ1,γ2)
		else
			OutCoeff[ixMN] = 0
		end
	end

	n = 0
	for m = 2:N+1-2n
		ixMN = pairing(m,n,true) # Index associated to (m,0,Even)
		ixMmN = pairing(m-1,n,false) # Index associated to (m-1,0,Odd)
		if m+2n≤N-1
			ixMmNp = pairing(m-1,n+1,false) # Index associated to (m-1,1,Odd)
			ixMpN = pairing(m+1,n,false) # Index associated to (m+1,n,Odd)
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2odd(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H2odd(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I2odd(m+1,n,α,γ1,γ2)
		else
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2odd(m-1,n,α,γ1,γ2)
		end
	end

	for m = 0:1
		for n = 1:(N+1-m)÷2
			ixMN = pairing(m,n,true) # Index associated to (m,n,Even)
			ixMpNm = pairing(m+1,n-1,false) # Index associated to (m+1,n-1,Odd)
			if m+2n≤N-1
				ixMpN = pairing(m+1,n,false) # Index associated to (m+1,n,Odd)
				OutCoeff[ixMN] = OrigCoeff[ixMpN]*I2odd(m+1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J2odd(m+1,n-1,α,γ1,γ2)
			else
				OutCoeff[ixMN] = OrigCoeff[ixMpNm]*J2odd(m+1,n-1,α,γ1,γ2)
			end
		end
	end

	for n = 1:(N+1)÷2
		for m = 2:N+1-2n
			ixMN = pairing(m,n,true) # Index associated to (m,n,Even)
			ixMmN = pairing(m-1,n,false) # Index associated to (m-1,n,Odd)
			ixMpNm = pairing(m+1,n-1,false) # Index associated to (m+1,n-1,Odd)
			if m+2n≤N-1
				ixMmNp = pairing(m-1,n+1,false) # Index associated to (m-1,n+1,Odd)
				ixMpN = pairing(m+1,n,false) # Index associated to (m+1,n,Odd)
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2odd(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H2odd(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I2odd(m+1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J2odd(m+1,n-1,α,γ1,γ2)
			else
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2odd(m-1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J2odd(m+1,n-1,α,γ1,γ2)
			end
		end
	end

	# Odd part

	for m = 1:N+1
		n = 0
		ixMN = pairing(m,n,false) # Index associated to (m,0,Odd)
		ixMmN = pairing(m-1,n,true) # Index associated to (m-1,0,Even)
		if m+2n≤N-1
			ixMmNp = pairing(m-1,n+1,true) # Index associated to (m-1,1,Even)
			ixMpN = pairing(m+1,n,true) # Index associated to (m+1,0,Even)
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2even(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H2even(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I2even(m+1,n,α,γ1,γ2)
		else
			OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2even(m-1,n,α,γ1,γ2)
		end
		for n = 1:(N+1-m)÷2
			ixMN = pairing(m,n,false) # Index associated to (m,n,Odd)
			ixMmN = pairing(m-1,n,true) # Index associated to (m-1,n,Even)
			ixMpNm = pairing(m+1,n-1,true) # Index associated to (m+1,n-1,Even)
			if m+2n≤N-1
				ixMmNp = pairing(m-1,n+1,true) # Index associated to (m-1,n+1,Even)
				ixMpN = pairing(m+1,n,true) # Index associated to (m+1,n,Even)
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2even(m-1,n,α,γ1,γ2) + OrigCoeff[ixMmNp]*H2even(m-1,n+1,α,γ1,γ2) + OrigCoeff[ixMpN]*I2even(m+1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J2even(m+1,n-1,α,γ1,γ2)
			else
				OutCoeff[ixMN] = OrigCoeff[ixMmN]*G2even(m-1,n,α,γ1,γ2) + OrigCoeff[ixMpNm]*J2even(m+1,n-1,α,γ1,γ2)
			end
		end
	end
	DZFun([γ1,γ2,α],N+1,OutCoeff)
end

function symx1(f::DZFun)
	outcoefs = deepcopy(f.coefficients)
	for i = 1:polyDim(f.degree)
		(m,n,even) = inversepairing(i)
		meven = iseven(m)
		if xor(even,meven)
			outcoefs[i] = 0.0
		end
	end
	DZFun(f.κ, f.degree, outcoefs)
end

function skewx1(f::DZFun)
	outcoefs = deepcopy(f.coefficients)
	for i = 1:polyDim(f.degree)
		(m,n,even) = inversepairing(i)
		meven = iseven(m)
		if ~xor(even,meven)
			outcoefs[i] = 0.0
		end
	end
	DZFun(f.κ, f.degree, outcoefs)
end

function symx2(f::DZFun)
	outcoefs = deepcopy(f.coefficients)
	for i = 1:polyDim(f.degree)
		(m,n,even) = inversepairing(i)
		if ~even
			outcoefs[i] = 0.0
		end
	end
	DZFun(f.κ, f.degree, outcoefs)
end

function skewx2(f::DZFun)
	outcoefs = deepcopy(f.coefficients)
	for i = 1:polyDim(f.degree)
		(m,n,even) = inversepairing(i)
		if even
			outcoefs[i] = 0.0
		end
	end
	DZFun(f.κ, f.degree, outcoefs)
end

"""
Compute the result of applying the angular Dunkl operator D_{12} to a DZFun without shifting parameters.
"""
DunklAngular(f::DZFun) = mbx1(Dunklx2(f)) - mbx2(Dunklx1(f))

"""
Compute the result of multiplying a DZFun by (1-x1^2-x2^2)
"""
mbr(f::DZFun) = f-mbx1(mbx1(f))-mbx2(mbx2(f))

"""
Compute the (α,γ)-adjoint of the Dunkl operator applied in a DZFun
"""
adjointDunklx1(f::DZFun,α::Float64) = -mbr(Dunklx1(f)) + 2*(α+1)*mbx1(f)
adjointDunklx2(f::DZFun,α::Float64) = -mbr(Dunklx2(f)) + 2*(α+1)*mbx2(f)
end # module
