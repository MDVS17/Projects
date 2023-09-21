Select * 
from Portfolio..CovidDeaths
where continent is not null
order by 3,4

--Select * 
--from Portfolio..CovidVaccinations
--order by 3,4

-- Select The data we are going to be using

Select Location, date, total_cases, new_cases, total_deaths, population
from Portfolio..CovidDeaths
order by 1,2


-- TOTAL CASES vs TOTAL DEATHS (Death ratio of covid infected people) by Country

SELECT Location, date, total_cases,total_deaths, (CONVERT(float, total_deaths) / NULLIF(CONVERT(float, total_cases), 0))*100 as Death_Ratio
from Portfolio..CovidDeaths
where location like '%Canada%'
order by 1,2


-- Percentage of Population got Covid
-- Total Cases vs Population

SELECT Location, date,population, total_cases, (CONVERT(float, total_cases) / NULLIF(CONVERT(float, population), 0))*100 as InfectedPercentage
from Portfolio..CovidDeaths
-- where location like '%Canada%'
order by 1,2


--Highest infected rate Countries
SELECT Location,population, MAX(cast(total_cases as int) as HighestInfectionCount, MAX((CONVERT(float, total_cases) / NULLIF(CONVERT(float, population), 0))*100)as PercentagePopulationInfected
from Portfolio..CovidDeaths
Group by Location,population
order by PercentagePopulationInfected desc


-- Countries with highest Death Count per population

SELECT Location, MAX(cast(total_deaths as int)) as TotalDeathCount
from Portfolio..CovidDeaths
where continent is not null
Group by Location
order by TotalDeathCount desc


-- Braking by Continent

SELECT continent, MAX(cast(total_deaths as int)) as TotalDeathCount
from Portfolio..CovidDeaths
where continent is not null
Group by continent
order by TotalDeathCount desc


-- Continents with the highest Death Counts

SELECT continent, MAX(cast(total_deaths as int)) as TotalDeathCount
from Portfolio..CovidDeaths
where continent is not null
Group by continent
order by TotalDeathCount desc


-- Global Numbers

Select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, SUM(cast(new_deaths as int))/SUM(New_Cases)*100 as DeathPercentage
From Portfolio..CovidDeaths
--Where location like '%states%'
where continent is not null 
--Group By date
order by 1,2


-- Looking at the total population vs Vaccinations

Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition by dea.Location order by dea.location, dea.Date) as CumulativeVaccination
, (CumulativeVaccination/population)*100
from Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date=vac.date
where dea.continent is not null
order by 2,3


-- USE CTE

With PopvsVasc (Continent, Location, Date, Population, New_Vaccinations, CumulativeVaccination)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition by dea.Location order by dea.location, dea.Date) as CumulativeVaccination
--, (CumulativeVaccination/population)*100
from Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date=vac.date
where dea.continent is not null
--order by 2,3
)
Select *, (CumulativeVaccination/Population)*100
from PopvsVasc


-- Temp Table
DROP table if exists #PercentPopulationVaccinated
Create table #PercentPopulationVaccinated
(
Continent nvarchar (255),
Location nvarchar (255),
Date datetime,
Population numeric,
New_vaccinations numeric,
CumulativeVaccination numeric
)


Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(Cast(vac.new_vaccinations as int)) OVER (Partition by dea.Location order by dea.location, dea.Date) as CumulativeVaccination
--, (CumulativeVaccination/population)*100
from Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date=vac.date
where dea.continent is not null
--order by 2,3

Select *, (CumulativeVaccination/Population)*100
from #PercentPopulationVaccinated



-- CREATING VIEW TO STORE DATA FOR THE VISUALIZATION

Create view PercentPopulationVaccinated as 
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition by dea.Location order by dea.location, dea.Date) as CumulativeVaccination
--, (CumulativeVaccination/population)*100
from Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	on dea.location = vac.location
	and dea.date=vac.date
where dea.continent is not null


Select * 
from PercentPopulationVaccinated