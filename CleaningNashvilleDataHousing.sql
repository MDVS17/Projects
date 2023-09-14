/* 
Cleaning Data 
*/

Select * 
from Portfolio.dbo.NashvilleHousing

-- Standarize date format


Select SaleDate, CONVERT(Date,SaleDate)
from Portfolio.dbo.NashvilleHousing

Update NashvilleHousing
SET SaleDate = CONVERT(Date,SaleDate)

ALTER TABLE NashvilleHousing
Add SaleDateConverted Date;

update	NashvilleHousing
SET SaleDateConverted = CONVERT(Date,SaleDate)



-- Populate property adress data


Select *
from Portfolio.dbo.NashvilleHousing
--Where PropertyAddress is null
order by ParcelID



Select a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress,ISNULL(a.PropertyAddress,b.PropertyAddress)
from Portfolio.dbo.NashvilleHousing a
JOIN Portfolio.dbo.NashvilleHousing b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID] <> b.[UniqueID]
WHERE a.PropertyAddress is null



Update a
SET PropertyAddress = ISNULL(a.PropertyAddress,b.PropertyAddress)
From Portfolio.dbo.NashvilleHousing a
JOIN Portfolio.dbo.NashvilleHousing b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID] <> b.[UniqueID]
WHERE a.PropertyAddress is null



-- Braking out address into individual columns (Address, City, State)


Select PropertyAddress
from Portfolio.dbo.NashvilleHousing
--Where PropertyAddress is null
--order by ParcelID


-- we have , as separator between the address and the city 

SELECT
SUBSTRING(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) -1 ) as Address
, SUBSTRING(PropertyAddress, CHARINDEX(',', PropertyAddress) +1, LEN(PropertyAddress)) as Address
From Portfolio.dbo.NashvilleHousing


-- We create the new Columns for the split address for city and the address


ALTER TABLE NashvilleHousing
Add ProperlySplitAddress Nvarchar(255);

update	NashvilleHousing
SET ProperlySplitAddress = SUBSTRING(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) -1 )


ALTER TABLE NashvilleHousing
Add ProperlySplitCity Nvarchar(255);

update	NashvilleHousing
SET ProperlySplitCity = SUBSTRING(PropertyAddress, CHARINDEX(',', PropertyAddress) +1, LEN(PropertyAddress))


Select *
From Portfolio.dbo.NashvilleHousing


-- Now we going to separate the Owner Address


Select OwnerAddress
From Portfolio.dbo.NashvilleHousing

-- With PARSAME function we could separate the text but the default separator is the '.' so we replace the ',' by '.'

Select
PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3)
,PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2)
, PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1)
From Portfolio.dbo.NashvilleHousing



-- now create the columns with the code above



ALTER TABLE NashvilleHousing
Add OwnerSplitAddress Nvarchar(255);

update	NashvilleHousing
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3)


ALTER TABLE NashvilleHousing
Add OwnerSplitCity Nvarchar(255);

update	NashvilleHousing
SET OwnerSplitCity = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2)

ALTER TABLE NashvilleHousing
Add OwnerSplitState Nvarchar(255);

update	NashvilleHousing
SET OwnerSplitState = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1)



Select OwnerSplitAddress,OwnerSplitCity,OwnerSplitState
From Portfolio.dbo.NashvilleHousing


-- Normalize the answer in "Sold as Vacant" field to have Yes and No


-- Start counting the different values 

Select Distinct(SoldAsVacant), count(SoldAsVacant)
From Portfolio.dbo.NashvilleHousing
Group by SoldAsVacant
order by 2


Select SoldAsVacant
, CASE  When SoldAsVacant = 'Y' then 'Yes'
		When SoldAsVacant = 'N' then 'No'
		else SoldAsVacant
		END
From Portfolio.dbo.NashvilleHousing


Update NashvilleHousing
SET SoldAsVacant = CASE  When SoldAsVacant = 'Y' then 'Yes'
		When SoldAsVacant = 'N' then 'No'
		else SoldAsVacant
		END



-- Remove Duplicates

WITH RowNumCTE AS(
Select *, 
	ROW_NUMBER() OVER(
	PARTITION BY ParcelID,
				PropertyAddress,
				SalePrice,
				SaleDate,
				LegalReference
				ORDER BY 
					UniqueID
					) row_num

From Portfolio.dbo.NashvilleHousing
-- order by ParcelID
)

SELECT *
From RowNumCTE
Where row_num > 1
order by PropertyAddress



-- Select Unused columns 

SELECT *
from Portfolio.dbo.NashvilleHousing


ALTER TABLE Portfolio.dbo.NashvilleHousing
DROP COLUMN OwnerAddress, TaxDistrict, PropertyAddress

ALTER TABLE Portfolio.dbo.NashvilleHousing
DROP COLUMN SaleDate