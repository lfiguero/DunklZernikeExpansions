using DunklZernikeExpansions
using LinearAlgebra

@assert map(DunklZernikeExpansions.polyDim, 0:5) == [1, 3, 6, 10, 15, 21]

maxdeg = 10
index_triplets = [(m,(deg-m)÷2,even) for deg=0:maxdeg for m=deg:-2:0 for even in (true,false) if (m>0 || even)]
indices = map(v -> DunklZernikeExpansions.pairing(v...), index_triplets)
@assert indices==1:DunklZernikeExpansions.polyDim(maxdeg)
@assert index_triplets == map(DunklZernikeExpansions.inversepairing, indices)

d = 4
v = randn(DunklZernikeExpansions.polyDim(d))
f1 = DZFun((1,2,3),d,v)
f2 = DZFun((1,2,3.0),v)
@assert f1 == f2
f3 = DZFun([1,2,3],v)
f4 = DZFun([1.0,2.0,3.0],[v;0.0;0.0;0.0;0.0])
f5 = DZFun([1.0,2.0,3.0],d,v)
@assert f3 == f4
@assert f3 == f5

g1 = DZPoly(DZParam(-1/2,-1/3,0),2,1,true)
g2 = DZPoly((-1/2,-1/3,0),2,1,true)
g3 = DZPoly([-1/2,-1/3,0],2,1,true)
@assert g1==g2
@assert g1==g3

points = [randn(2) for i=1:100]
parameters = [DZParam(10*rand(3)-[1.0,1.0,1.0]...) for i = 1:100]

# Test raise via evalDZ
d = 20
v = randn(DunklZernikeExpansions.polyDim(d))

for param in parameters 
	f = DZFun(param,d,v)
	rf = DunklZernikeExpansions.raise(f)
	for point in points
		@assert evalDZ(f,point[1],point[2]) ≈ evalDZ(rf,point[1],point[2])
	end
end

# Test lower via evalDZ
d = 20
v = randn(DunklZernikeExpansions.polyDim(d))
for param in parameters 
	shiftedparam = DZParam(param.γ1,param.γ2,param.α+1.0)
	f = DZFun(shiftedparam,d,v)
	lf = DunklZernikeExpansions.lower(f)
	for point in points
		@assert evalDZ(f,point[1],point[2]) ≈ evalDZ(lf,point[1],point[2])
	end
end

# Test whether lower and raise are the inverse of one another
for param in parameters
	f = DZFun(param,d,v)
	r = DunklZernikeExpansions.lower(DunklZernikeExpansions.raise(f))-f
	@assert norm(r.coefficients) < 1e-10
end
for param in parameters
	shiftedparam = DZParam(param.γ1,param.γ2,param.α+1.0)
	f = DZFun(shiftedparam,d,v)
	r = DunklZernikeExpansions.raise(DunklZernikeExpansions.lower(f))-f
	@assert norm(r.coefficients) < 1e-10
end

# Test commutativity between the Dunkl operators and raise and between the Dunkl operators
for param in parameters
	f = DZFun(param,d,v)
	
	Dx1R = Dunklx1(DunklZernikeExpansions.raise(f))
	RDx1 = DunklZernikeExpansions.raise(Dunklx1(f))
	Dx2R = Dunklx2(DunklZernikeExpansions.raise(f))
	RDx2 = DunklZernikeExpansions.raise(Dunklx2(f))
	Dx1Dx2 = Dunklx1(Dunklx2(f))
	Dx2Dx1 = Dunklx2(Dunklx1(f))

	for point in points
		@assert evalDZ(Dx1R,point[1],point[2]) ≈ evalDZ(RDx1,point[1],point[2])
		@assert evalDZ(Dx2R,point[1],point[2]) ≈ evalDZ(RDx2,point[1],point[2])
		@assert evalDZ(Dx1Dx2,point[1],point[2]) ≈ evalDZ(Dx2Dx1,point[1],point[2])
	end
end

# Test mbx1 and mbx2 via evalDZ
for param in parameters
	f = DZFun(param,d,v)
	fbx1 = mbx1(f)
	fbx2 = mbx2(f)

	for point in points
		@assert evalDZ(fbx1,point[1],point[2]) ≈ point[1]*evalDZ(f,point[1],point[2])
		@assert evalDZ(fbx2,point[1],point[2]) ≈ point[2]*evalDZ(f,point[1],point[2])
	end
end

# Test sym and skew operators
for param in parameters
	f = DZFun(param,d,v)
	for point in points
		val = evalDZ(f,point[1],point[2])
		valsigma1star = evalDZ(f,-point[1],point[2])
		valsigma2star = evalDZ(f,point[1],-point[2])
		@assert evalDZ(symx1(f),point[1],point[2]) ≈ (val+valsigma1star)/2.0
		@assert evalDZ(skewx1(f),point[1],point[2]) ≈ (val-valsigma1star)/2.0
		@assert evalDZ(symx2(f),point[1],point[2]) ≈ (val+valsigma2star)/2.0
		@assert evalDZ(skewx2(f),point[1],point[2]) ≈ (val-valsigma2star)/2.0
	end
end

# Test of Sturm–Liouville problem satisfied by Lebesgue orthogonal polynomials
function SL(f::DZFun,α::Real)
	γ1 = f.κ.γ1; γ2 = f.κ.γ2
	generalizedMinusDivGrad = adjointDunklx1(Dunklx1(f),α) + adjointDunklx2(Dunklx2(f),α)
	DunklLaplaceBeltrami = DunklAngular(DunklAngular(f))
	skewsA = (2*α+γ1+γ2+2.0)*(γ1*skewx1(f)+γ2*skewx2(f))
	skewsB = γ1*γ1*skewx1(f) + 2*γ1*γ2*skewx1(skewx2(f)) + γ2*γ2*skewx2(f)
	generalizedMinusDivGrad - DunklLaplaceBeltrami - skewsA + skewsB
end
for param in parameters
	for i = 1:200
		(m,n,even) = DunklZernikeExpansions.inversepairing(i)
		p = DZPoly(param, m, n, even)
		Lp = SL(p,p.κ.α)
		theoreticalLp = p.degree*(p.degree + 2*param.α + param.γ1 + param.γ2 + 2.0)*p
		@assert Lp ≈ theoreticalLp
	end
end

function M(f::DZFun,α::Real)
	adjointDunklx1(adjointDunklx1(f,α),α-1) + adjointDunklx2(adjointDunklx2(f,α),α-1)
end

# Test of Sturm–Liouville problem satisfied by Dunkl-Sobolev orthogonal polynomials

for param in parameters
	for deg = 0:1
		v = rand(DunklZernikeExpansions.polyDim(deg) - DunklZernikeExpansions.polyDim(deg-1))
		v = [zeros(DunklZernikeExpansions.polyDim(deg-1)) ; v]
		p = DZFun(param,v)
		@assert p.degree == deg
		Lp = SL(p,p.κ.α-1)
		theoreticalLp = p.degree*(p.degree + 2*(param.α-1.) + param.γ1 + param.γ2 + 2.0)*p
		@assert Lp ≈ theoreticalLp
	end
	for deg = 3:20
		vh = rand(DunklZernikeExpansions.polyDim(deg) - DunklZernikeExpansions.polyDim(deg-1))
		vh = [zeros(DunklZernikeExpansions.polyDim(deg-1)) ; vh]
		for i = DunklZernikeExpansions.polyDim(deg-1)+1:length(vh)
			if DunklZernikeExpansions.inversepairing(i)[2] != 0
				vh[i] = 0.
			end
		end
		ph = DZFun(param,vh) #harmonic part

		vm = rand(DunklZernikeExpansions.polyDim(deg-2) - DunklZernikeExpansions.polyDim(deg-3))
		vm = [zeros(DunklZernikeExpansions.polyDim(deg-3)) ; vm]
		pm = DZFun([param.γ1,param.γ2,param.α+1.],vm)
		pm = DunklZernikeExpansions.lower(M(pm,param.α)) #M part

		p = ph + pm
		@assert p.degree == deg
		Lp = SL(p,p.κ.α-1)
		theoreticalLp = p.degree*(p.degree + 2*(param.α-1.) + param.γ1 + param.γ2 + 2.0)*p
		@assert Lp ≈ theoreticalLp
	end
end

# Test commutation property

for param in parameters
	f = DZFun(param,d,v)
	α = randn()
	Dunklx1(SL(f,α-1)) - SL(Dunklx1(f),α) ≈ (2*α+f.κ.γ1+f.κ.γ2+2-1)*Dunklx1(f)
end