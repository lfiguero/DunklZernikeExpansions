export veryParticularSequence, extendedMapSequence

function sip(f::DZFun, g::DZFun)
	@assert f.κ ≈ g.κ
	semisip = wip(Dunklx1(f),Dunklx1(g))+wip(Dunklx2(f),Dunklx2(g))
	semisip + wip(project(f, 0), project(g,0))
end

function Hessiansemiip(f::DZFun, g::DZFun)
	@assert f.κ ≈ g.κ
	d1f = Dunklx1(f); d11f = Dunklx1(d1f)
	d2f = Dunklx2(f); d22f = Dunklx2(d2f)
	d12f = Dunklx1(d2f)
	d1g = Dunklx1(g); d11g = Dunklx1(d1g)
	d2g = Dunklx2(g); d22g = Dunklx2(d2g)
	d12g = Dunklx1(d2g)
	wip(d11f,d11g) + 2.0*wip(d12f,d12g) + wip(d22f,d22g)
end

function veryParticularSequence(κ::DZParam, maxn::Int64)
	@assert maxn ≥ 6
	op = [lower(DZPoly(DZParam(κ.γ1,κ.γ2,κ.α+1.0), 0, n, true)) for n=4:maxn]
	sop = [adjointDunklx1(adjointDunklx1(p,κ.α),κ.α-1)+adjointDunklx2(adjointDunklx2(p,κ.α),κ.α-1) for p in op]
	c = [-Hessiansemiip(sop[i+1],sop[i])/Hessiansemiip(sop[i],sop[i]) for i = 1:length(sop)-1]
	t = [c[i]*sop[i] + sop[i+1] for i = 1:length(sop)-1]
	projectionDegree = [2*n+1 for n=4:maxn-1]
	res = sop[2:end]
	seminormRatio = [sqrt(Hessiansemiip(res[i],res[i])/Hessiansemiip(t[i],t[i])) for i = 1:length(sop)-1]
	empiricalRate = log.(seminormRatio[2:end]./seminormRatio[1:end-1]) ./ log.(projectionDegree[2:end]./projectionDegree[1:end-1])
end

# Criminally inefficient H¹ projection
function sipproject(f::DZFun, N::Int64)
	if f.degree <= N
		f
	else
		basis = [DZPoly(f.κ, inversepairing(i)...) for i=1:polyDim(N)]
		Gram = [sip(p,q) for p in basis, q in basis]
		rhs = [sip(p,f) for p in basis]
		coefs = Gram\rhs
		sum(coefs .* basis)
	end
end

# I lose patience with this for maxn > 15
function extendedMapSequence(κ::DZParam, maxn::Int64)
	@assert maxn ≥ 6
	op = [DZPoly(κ, 0, n, true) for n=5:maxn]
	t = [adjointDunklx1(adjointDunklx1(p,κ.α-1),κ.α-2)+adjointDunklx2(adjointDunklx2(p,κ.α-1),κ.α-2) for p in op]
	projectionDegree = [2*n+1 for n=5:maxn]
	projections = [sipproject(t[i],projectionDegree[i]) for i=1:length(t)]
	res = [t[i] - projections[i] for i=1:length(t)]
	seminormRatio = [sqrt(Hessiansemiip(res[i],res[i])/Hessiansemiip(t[i],t[i])) for i = 1:length(t)]
	empiricalRate = log.(seminormRatio[2:end]./seminormRatio[1:end-1]) ./ log.(projectionDegree[2:end]./projectionDegree[1:end-1])
end
